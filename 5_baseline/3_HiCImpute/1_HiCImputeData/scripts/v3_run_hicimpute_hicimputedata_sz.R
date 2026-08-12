#!/usr/bin/env Rscript

# Run HiCImpute MCMCImpute with niter=5000, burnin=1000 and save Impute_SZ
# (structural-zero thresholded output) in numpy row-major tril order,
# matching the legacy npz_dxy format.

argval <- function(args, key, default = NULL) {
  idx <- which(args == key)
  if (length(idx) == 0L) return(default)
  if (idx == length(args)) stop(paste("missing value for", key))
  args[idx + 1L]
}

main <- function() {
  suppressPackageStartupMessages(library(HiCImpute))

  args <- commandArgs(trailingOnly = TRUE)
  dataset <- argval(args, "--dataset")
  input_root <- argval(args, "--input-root")
  output_root <- argval(args, "--output-root")
  n_beads <- as.integer(argval(args, "--n-beads", "61"))
  niter <- as.integer(argval(args, "--niter", "5000"))
  burnin <- as.integer(argval(args, "--burnin", "1000"))
  mc_cores <- as.integer(argval(args, "--mc-cores", "1"))
  seed <- as.integer(argval(args, "--seed", "1234"))
  python_bin <- argval(args, "--python", "")
  stopifnot(!is.null(dataset), !is.null(input_root), !is.null(output_root))

  n_features <- as.integer(n_beads * (n_beads - 1L) / 2L)
  in_dir <- file.path(input_root, dataset)
  schic_file <- file.path(in_dir, "schic.bin")
  bulk_file <- file.path(in_dir, "bulk.bin")
  order_file <- file.path(in_dir, "feature_order.npy")
  stopifnot(file.exists(schic_file), file.exists(bulk_file), file.exists(order_file))

  file_bytes <- file.size(schic_file)
  n_cells <- as.integer(file_bytes / (8 * n_features))
  stopifnot(file_bytes == 8 * n_features * n_cells)

  cat(sprintf("[hicimpute-sz] %s: reading %d x %d (features x cells)\n", dataset, n_features, n_cells))
  schic <- matrix(readBin(schic_file, "double", n = n_features * n_cells, endian = "little"),
                  nrow = n_features, ncol = n_cells)
  bulk <- readBin(bulk_file, "double", n = n_features, endian = "little")

  if (sum(bulk) <= 0) stop("bulk sum must be positive")

  if (nchar(python_bin) == 0L) python_bin <- "python3"
  read_order_py <- tempfile(pattern = "read_order_", fileext = ".py")
  on.exit(unlink(read_order_py), add = TRUE)
  writeLines(sprintf(
    "import numpy as np\no = np.load(%s)\nprint(' '.join(map(str, o.tolist())));",
    shQuote(order_file, type = "sh")
  ), read_order_py)
  order_str <- system2(python_bin, read_order_py, stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(order_str, "status")) && attr(order_str, "status") != 0L) {
    cat("Python stderr:\n")
    writeLines(order_str, stderr())
    stop("reading feature_order.npy failed")
  }
  order <- as.integer(unlist(strsplit(trimws(paste(order_str, collapse = " ")), "\\s+")))
  if (length(order) != n_features) stop(sprintf("feature_order length %d != n_features %d", length(order), n_features))

  inv_order <- integer(length(order))
  inv_order[order + 1L] <- seq_len(n_features)

  set.seed(seed)
  startval <- c(100, 100, 10, 8, 10, 0.1, 900, 0.2, 0, rep(8, n_cells))

  cat(sprintf("[hicimpute-sz] %s: MCMCImpute n=%d niter=%d burnin=%d mc.cores=%d (Impute_SZ)\n",
              dataset, n_beads, niter, burnin, mc_cores))
  t0 <- proc.time()
  result <- MCMCImpute(
    scHiC = schic,
    bulk = bulk,
    startval = startval,
    n = n_beads,
    mc.cores = mc_cores,
    cutoff = 0.5,
    niter = niter,
    burnin = burnin
  )
  cat(sprintf("[hicimpute-sz] %s: MCMCImpute done in %.1f min\n", dataset, (proc.time() - t0)["elapsed"] / 60))

  # Use Impute_SZ (structural-zero thresholded) instead of Impute_All
  impute_sz_r <- result$Impute_SZ
  impute_sz_numpy <- impute_sz_r[inv_order, , drop = FALSE]
  pred_cells_by_features <- t(impute_sz_numpy)

  npz_path <- file.path(output_root, "npz_dxy", paste0(dataset, "_niter", niter, "_burnin", burnin, ".npz"))
  dir.create(dirname(npz_path), recursive = TRUE, showWarnings = FALSE)
  tmp_bin <- tempfile(pattern = "pred_", fileext = ".bin")
  writeBin(as.double(as.vector(t(pred_cells_by_features))), tmp_bin, size = 8L, endian = "little")
  on.exit(unlink(tmp_bin), add = TRUE)

  py_script <- sprintf(
    paste0(
      "import numpy as np\n",
      "from scipy.sparse import coo_matrix, save_npz\n",
      "v = np.fromfile(%s, dtype=np.float64)\n",
      "v = v.reshape((%d, %d))\n",
      "save_npz(%s, coo_matrix(v).tocsr())\n"
    ),
    shQuote(tmp_bin, type = "sh"), n_cells, n_features, shQuote(npz_path, type = "sh")
  )
  py_tmp <- tempfile(pattern = "write_npz_", fileext = ".py")
  writeLines(py_script, py_tmp)
  on.exit(unlink(py_tmp), add = TRUE)
  rc <- system2(python_bin, py_tmp, stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(rc, "status")) && attr(rc, "status") != 0L) {
    cat("NPZ write failed:\n")
    writeLines(rc, stderr())
    stop("NPZ write failed")
  }

  cat(sprintf("[hicimpute-sz] %s: wrote %s\n", dataset, npz_path))
  cat(sprintf("[hicimpute-sz] %s: done\n", dataset))
}

main()