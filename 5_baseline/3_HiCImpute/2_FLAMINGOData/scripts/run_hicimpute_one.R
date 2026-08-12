#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(HiCImpute))

parse_args <- function(argv) {
  out <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key))
    }
    name <- substring(key, 3)
    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) {
      out[[name]] <- TRUE
      i <- i + 1
    } else {
      out[[name]] <- argv[[i + 1]]
      i <- i + 2
    }
  }
  out
}

required_arg <- function(args, name) {
  value <- args[[name]]
  if (is.null(value)) {
    stop(sprintf("Missing required --%s", name))
  }
  value
}

read_f64_matrix <- function(path, nrow, ncol) {
  con <- file(path, "rb")
  on.exit(close(con))
  values <- readBin(con, what = "double", n = nrow * ncol, size = 8, endian = "little")
  if (length(values) != nrow * ncol) {
    stop(sprintf("%s has %d values; expected %d", path, length(values), nrow * ncol))
  }
  matrix(values, nrow = nrow, ncol = ncol)
}

read_f64_vector <- function(path, n) {
  con <- file(path, "rb")
  on.exit(close(con))
  values <- readBin(con, what = "double", n = n, size = 8, endian = "little")
  if (length(values) != n) {
    stop(sprintf("%s has %d values; expected %d", path, length(values), n))
  }
  values
}

write_f64_matrix <- function(path, matrix_value) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.double(matrix_value), con, size = 8, endian = "little")
}

argv <- commandArgs(trailingOnly = TRUE)
args <- parse_args(argv)

dataset_id <- required_arg(args, "dataset")
input_root <- normalizePath(required_arg(args, "input-root"), mustWork = TRUE)
output_root <- required_arg(args, "output-root")
n_beads <- as.integer(required_arg(args, "n-beads"))
n_features <- as.integer(required_arg(args, "n-features"))
n_cells <- as.integer(required_arg(args, "n-cells"))
niter <- as.integer(ifelse(is.null(args[["niter"]]), "5000", args[["niter"]]))
burnin <- as.integer(ifelse(is.null(args[["burnin"]]), "1000", args[["burnin"]]))
mc_cores <- as.integer(ifelse(is.null(args[["mc-cores"]]), "1", args[["mc-cores"]]))
seed <- as.integer(ifelse(is.null(args[["seed"]]), "1234", args[["seed"]]))

if (n_features != n_beads * (n_beads - 1) / 2) {
  stop(sprintf("n_features=%d does not match n_beads=%d", n_features, n_beads))
}

input_dir <- file.path(input_root, dataset_id)
sc_path <- file.path(input_dir, "schic_features_by_cells.bin")
bulk_path <- file.path(input_dir, "bulk_vector.bin")
expected_path <- file.path(input_dir, "expected_features_by_cells.bin")

sc_hic <- read_f64_matrix(sc_path, nrow = n_features, ncol = n_cells)
bulk <- read_f64_vector(bulk_path, n_features)
expected <- if (file.exists(expected_path)) {
  read_f64_matrix(expected_path, nrow = n_features, ncol = n_cells)
} else {
  NULL
}

single_sum <- colSums(sc_hic)
if (any(single_sum <= 0)) {
  stop(sprintf("Dataset %s contains cells with zero total contacts", dataset_id))
}
if (sum(bulk) <= 0) {
  stop(sprintf("Dataset %s has zero bulk contacts", dataset_id))
}

set.seed(seed)
startval <- c(100, 100, 10, 8, 10, 0.1, 900, 0.2, 0, rep(8, n_cells))
message(sprintf(
  "Running HiCImpute dataset=%s n_beads=%d n_cells=%d niter=%d burnin=%d mc.cores=%d",
  dataset_id, n_beads, n_cells, niter, burnin, mc_cores
))

result <- MCMCImpute(
  scHiC = sc_hic,
  bulk = bulk,
  expected = expected,
  startval = startval,
  n = n_beads,
  mc.cores = mc_cores,
  cutoff = 0.5,
  niter = niter,
  burnin = burnin
)

rds_dir <- file.path(output_root, "rds")
bin_dir <- file.path(output_root, "bin")
dir.create(rds_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(bin_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(
  list(
    dataset_id = dataset_id,
    n_beads = n_beads,
    n_features = n_features,
    n_cells = n_cells,
    niter = niter,
    burnin = burnin,
    mc_cores = mc_cores,
    seed = seed,
    result = result
  ),
  file = file.path(rds_dir, sprintf("%s_hicimpute_result.rds", dataset_id))
)
write_f64_matrix(file.path(bin_dir, sprintf("%s_Impute_All.bin", dataset_id)), result$Impute_All)
write_f64_matrix(file.path(bin_dir, sprintf("%s_Impute_SZ.bin", dataset_id)), result$Impute_SZ)

summary_path <- file.path(output_root, sprintf("%s_run_summary.tsv", dataset_id))
write.table(
  data.frame(
    dataset_id = dataset_id,
    n_beads = n_beads,
    n_features = n_features,
    n_cells = n_cells,
    niter = niter,
    burnin = burnin,
    mc_cores = mc_cores,
    seed = seed,
    stringsAsFactors = FALSE
  ),
  file = summary_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
message(sprintf("Wrote %s", file.path(rds_dir, sprintf("%s_hicimpute_result.rds", dataset_id))))
