#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(igraph)
  library(ggplot2)
})


parse_named_args <- function(args) {
  if (length(args) %% 2 != 0) {
    stop("Arguments must be provided as --name value pairs")
  }
  output <- list()
  for (i in seq(1, length(args), by = 2)) {
    key <- sub("^--", "", args[[i]])
    output[[key]] <- args[[i + 1]]
  }
  required <- c("embedding", "labels", "output-dir", "method")
  missing <- required[!required %in% names(output)]
  if (length(missing) > 0) {
    stop(sprintf("Missing arguments: %s", paste(missing, collapse = ", ")))
  }
  output
}


validate_cell_order <- function(expected, observed, context) {
  if (length(expected) != length(observed)) {
    stop(sprintf(
      "%s: expected %d cells but observed %d",
      context,
      length(expected),
      length(observed)
    ))
  }
  mismatch <- which(expected != observed)
  if (length(mismatch) > 0) {
    i <- mismatch[[1]]
    stop(sprintf(
      "%s: cell order mismatch at row %d: expected %s, observed %s",
      context,
      i,
      expected[[i]],
      observed[[i]]
    ))
  }
}


cluster_levels <- function(values) {
  unique_values <- unique(as.character(values))
  numeric_values <- suppressWarnings(as.numeric(unique_values))
  if (all(is.finite(numeric_values))) {
    unique_values[order(numeric_values)]
  } else {
    sort(unique_values)
  }
}


composition_table <- function(metadata, variable) {
  values <- as.character(metadata[[variable]])
  clusters <- as.character(metadata$cluster)
  keep <- !is.na(values) & nzchar(values)
  values <- values[keep]
  clusters <- clusters[keep]
  observed <- as.data.frame(table(cluster = clusters, label = values), stringsAsFactors = FALSE)
  names(observed)[names(observed) == "Freq"] <- "count"
  observed <- observed[observed$count > 0, , drop = FALSE]

  all_combinations <- expand.grid(
    cluster = cluster_levels(metadata$cluster),
    label = sort(unique(values)),
    stringsAsFactors = FALSE
  )
  result <- merge(
    all_combinations,
    observed,
    by = c("cluster", "label"),
    all.x = TRUE,
    sort = FALSE
  )
  result$count[is.na(result$count)] <- 0L
  cluster_sizes <- as.data.frame(table(cluster = clusters), stringsAsFactors = FALSE)
  names(cluster_sizes)[names(cluster_sizes) == "Freq"] <- "cluster_size"
  result <- merge(result, cluster_sizes, by = "cluster", all.x = TRUE, sort = FALSE)
  result$proportion <- result$count / result$cluster_size
  result$variable <- variable
  result <- result[, c(
    "cluster", "label", "count", "cluster_size", "proportion", "variable"
  )]
  result$cluster <- factor(result$cluster, levels = cluster_levels(metadata$cluster))
  result <- result[order(result$cluster, result$label), , drop = FALSE]
  result$cluster <- as.character(result$cluster)
  rownames(result) <- NULL
  result
}


plot_composition_heatmap <- function(composition, variable, output_dir) {
  composition$cluster <- factor(
    composition$cluster,
    levels = rev(cluster_levels(composition$cluster))
  )
  composition$label <- factor(composition$label, levels = sort(unique(composition$label)))
  plot <- ggplot(composition, aes(x = label, y = cluster, fill = proportion)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "#2166AC", limits = c(0, 1)) +
    labs(x = NULL, y = "Louvain cluster", fill = "Within-cluster\nproportion") +
    theme_bw(base_size = 9) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_blank()
    )
  width <- max(6, min(14, 0.35 * length(unique(composition$label)) + 3))
  height <- max(4, min(12, 0.28 * length(unique(composition$cluster)) + 2))
  prefix <- file.path(output_dir, sprintf("%s_cluster_composition_heatmap", variable))
  ggsave(paste0(prefix, ".pdf"), plot, width = width, height = height, units = "in")
  ggsave(paste0(prefix, ".png"), plot, width = width, height = height, units = "in", dpi = 300)
}


paper_celltype_colors <- function(values) {
  known <- c(
    "Mitotic cell" = "#BEBEBE",
    "mitosis" = "#BEBEBE",
    "Blood" = "#FF9400",
    "blood" = "#FF9400",
    "ExE endoderm" = "#FEC44F",
    "ExE ectoderm" = "#CAB2D6",
    "EPI" = "#ADDD8E",
    "epiblast and PS" = "#ADDD8E",
    "Neural ectoderm" = "#AECBE6",
    "neural ectoderm" = "#AECBE6",
    "NMP" = "#96B9DB",
    "Neural tube" = "#7EA8D0",
    "neural tube" = "#7EA8D0",
    "Notochord" = "#6696C6",
    "notochord" = "#6696C6",
    "Radial glia" = "#4F85BB",
    "radial glias" = "#4F85BB",
    "OPC" = "#3773B1",
    "oligodendrocytes and progenitors" = "#3773B1",
    "Early neuron" = "#1F62A6",
    "early neurons" = "#1F62A6",
    "Schwann cell precursor" = "#08519C",
    "schwann cell precursors" = "#08519C",
    "Early mesoderm" = "#FC9272",
    "early mesoderm" = "#FC9272",
    "ExE mesoderm" = "#EF7F64",
    "Early mesenchyme" = "#E36C57",
    "early mesenchyme" = "#E36C57",
    "Intermediate mesoderm" = "#D6594A",
    "intermediate mesoderm" = "#D6594A",
    "Myocyte" = "#CA473C",
    "myocytes" = "#CA473C",
    "Mixed late mesenchyme" = "#BD342F",
    "mix late mesenchyme" = "#BD342F",
    "Endoderm" = "#FE9929",
    "endoderm" = "#FE9929",
    "Epithelial cell" = "#FA9FB5",
    "epithelial cells" = "#FA9FB5"
  )
  values <- sort(unique(as.character(values)))
  missing <- values[!values %in% names(known)]
  if (length(missing) > 0) {
    fallback <- grDevices::hcl.colors(length(missing), palette = "Dark 3")
    names(fallback) <- missing
    known <- c(known, fallback)
  }
  known[values]
}


plot_umaps <- function(umap_metadata, output_dir) {
  colors <- paper_celltype_colors(umap_metadata$celltype)
  common_theme <- theme_bw(base_size = 9) +
    theme(
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_blank(),
      panel.border = element_rect(linewidth = 0.4),
      legend.title = element_blank()
    )

  main_plot <- ggplot(
    umap_metadata,
    aes(x = schUMAP_1, y = schUMAP_2, color = celltype)
  ) +
    geom_point(size = 0.05, alpha = 0.9) +
    scale_color_manual(values = colors) +
    common_theme +
    labs(x = "scHiCluster UMAP 1", y = "scHiCluster UMAP 2")
  ggsave(
    file.path(output_dir, "celltype_umap.pdf"),
    main_plot,
    width = 5,
    height = 4,
    units = "in"
  )
  ggsave(
    file.path(output_dir, "celltype_umap.png"),
    main_plot,
    width = 5,
    height = 4,
    units = "in",
    dpi = 300
  )

  stages <- unique(as.character(umap_metadata$stage))
  split_plot <- main_plot +
    facet_wrap(~stage, nrow = 1) +
    theme(legend.position = "none")
  split_width <- max(8, 2.2 * length(stages))
  ggsave(
    file.path(output_dir, "celltype_umap_split_stage.pdf"),
    split_plot,
    width = split_width,
    height = 2.4,
    units = "in"
  )
  ggsave(
    file.path(output_dir, "celltype_umap_split_stage.png"),
    split_plot,
    width = split_width,
    height = 2.4,
    units = "in",
    dpi = 300
  )
}


calculate_modularity <- function(graph, clusters, resolution = 1.9) {
  graph_matrix <- methods::as(graph, "dgCMatrix")
  graph_matrix <- (graph_matrix + Matrix::t(graph_matrix)) / 2
  graph_matrix <- Matrix::drop0(graph_matrix)
  igraph_graph <- igraph::graph_from_adjacency_matrix(
    graph_matrix,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE
  )
  membership <- as.integer(factor(clusters, levels = cluster_levels(clusters)))
  modularity_value <- igraph::modularity(
    igraph_graph,
    membership = membership,
    weights = igraph::E(igraph_graph)$weight,
    resolution = resolution,
    directed = FALSE
  )
  data.frame(
    resolution = resolution,
    final_membership_modularity = as.numeric(modularity_value),
    n_nodes = igraph::vcount(igraph_graph),
    n_edges = igraph::ecount(igraph_graph),
    n_clusters = length(unique(clusters))
  )
}


extract_optimizer_modularity <- function(output_lines) {
  matched <- grep(
    "^Maximum modularity in [0-9]+ random starts:",
    output_lines,
    value = TRUE
  )
  if (length(matched) != 1L) {
    stop(sprintf(
      "Expected one optimizer modularity line, found %d",
      length(matched)
    ))
  }
  value <- sub("^.*: *", "", matched[[1]])
  value <- suppressWarnings(as.numeric(value))
  if (!is.finite(value)) {
    stop(sprintf("Could not parse optimizer modularity from: %s", matched[[1]]))
  }
  value
}


main <- function() {
  args <- parse_named_args(commandArgs(trailingOnly = TRUE))
  embedding_path <- normalizePath(args[["embedding"]], mustWork = TRUE)
  labels_path <- normalizePath(args[["labels"]], mustWork = TRUE)
  output_dir <- args[["output-dir"]]
  method <- args[["method"]]
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  embedding_table <- read.delim(
    embedding_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  labels <- read.csv(labels_path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!"cell_id" %in% colnames(labels)) {
    if (!"cellname" %in% colnames(labels)) {
      stop("Labels must contain cell_id or cellname")
    }
    labels$cell_id <- labels$cellname
  }
  required_labels <- c("cell_id", "stage", "celltype")
  missing_labels <- required_labels[!required_labels %in% colnames(labels)]
  if (length(missing_labels) > 0) {
    stop(sprintf("Labels are missing columns: %s", paste(missing_labels, collapse = ", ")))
  }
  svd_columns <- sprintf("SVD_%d", 1:20)
  missing_svd <- svd_columns[!svd_columns %in% colnames(embedding_table)]
  if (length(missing_svd) > 0) {
    stop(sprintf("Embedding is missing columns: %s", paste(missing_svd, collapse = ", ")))
  }
  if (anyDuplicated(labels$cell_id) || anyDuplicated(embedding_table$cell_id)) {
    stop("cell_id must be unique in labels and embedding")
  }
  validate_cell_order(labels$cell_id, embedding_table$cell_id, "Seurat input")

  metadata <- labels
  rownames(metadata) <- metadata$cell_id
  counts <- Matrix::sparseMatrix(
    i = rep(1L, nrow(metadata)),
    j = seq_len(nrow(metadata)),
    x = rep(1, nrow(metadata)),
    dims = c(1L, nrow(metadata)),
    dimnames = list("placeholder_contact_feature", metadata$cell_id)
  )
  object <- CreateSeuratObject(
    counts = counts,
    assay = "RNA",
    meta.data = metadata,
    project = method
  )

  embedding <- as.matrix(embedding_table[, svd_columns, drop = FALSE])
  storage.mode(embedding) <- "double"
  rownames(embedding) <- embedding_table$cell_id
  colnames(embedding) <- sprintf("SCH_%d", seq_len(ncol(embedding)))
  object[["schiclustersvd"]] <- CreateDimReducObject(
    embeddings = embedding,
    key = "SCH_",
    assay = "RNA"
  )

  object <- FindNeighbors(
    object,
    reduction = "schiclustersvd",
    dims = 1:15,
    graph.name = c("sch.nn", "sch.snn"),
    verbose = TRUE
  )
  object <- RunUMAP(
    object,
    reduction = "schiclustersvd",
    dims = 1:15,
    reduction.name = "schiclusterumap",
    reduction.key = "schUMAP_",
    seed.use = 42,
    verbose = TRUE
  )
  optimizer_output <- capture.output({
    object <- FindClusters(
      object,
      graph.name = "sch.nn",
      algorithm = 1,
      resolution = 1.9,
      random.seed = 0,
      verbose = TRUE
    )
  })
  cat(optimizer_output, sep = "\n")
  cat("\n")
  writeLines(optimizer_output, file.path(output_dir, "louvain_optimizer.log"))
  optimizer_max_modularity <- extract_optimizer_modularity(optimizer_output)

  clusters <- as.character(Idents(object))
  names(clusters) <- colnames(object)
  object$cluster <- clusters[colnames(object)]
  saveRDS(object, file.path(output_dir, "seurat_object.rds"))

  umap <- Embeddings(object, reduction = "schiclusterumap")
  umap_table <- data.frame(
    cell_id = rownames(umap),
    schUMAP_1 = umap[, 1],
    schUMAP_2 = umap[, 2],
    stringsAsFactors = FALSE
  )
  umap_table <- umap_table[match(labels$cell_id, umap_table$cell_id), , drop = FALSE]
  cluster_table <- data.frame(
    cell_id = labels$cell_id,
    cluster = clusters[labels$cell_id],
    stringsAsFactors = FALSE
  )
  write.table(
    umap_table,
    file.path(output_dir, "umap_coordinates.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  write.table(
    cluster_table,
    file.path(output_dir, "louvain_clusters.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )

  modularity <- calculate_modularity(
    object[["sch.nn"]],
    cluster_table$cluster,
    resolution = 1.9
  )
  modularity$optimizer_max_modularity <- optimizer_max_modularity
  modularity <- modularity[, c(
    "resolution",
    "optimizer_max_modularity",
    "final_membership_modularity",
    "n_nodes",
    "n_edges",
    "n_clusters"
  )]
  write.table(
    modularity,
    file.path(output_dir, "modularity.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )

  plot_metadata <- merge(labels, umap_table, by = "cell_id", sort = FALSE)
  plot_metadata <- plot_metadata[match(labels$cell_id, plot_metadata$cell_id), , drop = FALSE]
  plot_umaps(plot_metadata, output_dir)

  composition_metadata <- labels
  composition_metadata$cluster <- cluster_table$cluster
  composition_variables <- "celltype"
  if ("cellcycle_threshold" %in% colnames(labels)) {
    if (any(!is.na(labels$cellcycle_threshold) & nzchar(labels$cellcycle_threshold))) {
      composition_variables <- c(composition_variables, "cellcycle_threshold")
    }
  }
  for (variable in composition_variables) {
    composition <- composition_table(composition_metadata, variable)
    write.table(
      composition,
      file.path(output_dir, sprintf("%s_cluster_composition.tsv", variable)),
      sep = "\t",
      quote = FALSE,
      row.names = FALSE
    )
    plot_composition_heatmap(composition, variable, output_dir)
  }

  metadata_output <- list(
    method = method,
    n_cells = nrow(labels),
    input_embedding_dimensions = 20L,
    neighbor_dimensions = 1:15,
    umap_dimensions = 1:15,
    umap_reduction = "schiclusterumap",
    neighbor_graph = "sch.nn",
    clustering_algorithm = "Louvain",
    clustering_resolution = 1.9,
    modularity = modularity,
    composition_variables = composition_variables,
    package_versions = list(
      R = as.character(getRversion()),
      Seurat = as.character(packageVersion("Seurat")),
      SeuratObject = as.character(packageVersion("SeuratObject")),
      igraph = as.character(packageVersion("igraph")),
      ggplot2 = as.character(packageVersion("ggplot2"))
    )
  )
  saveRDS(metadata_output, file.path(output_dir, "seurat_run_metadata.rds"))
  cat(sprintf("Seurat outputs written to %s\n", normalizePath(output_dir)))
}


main()
