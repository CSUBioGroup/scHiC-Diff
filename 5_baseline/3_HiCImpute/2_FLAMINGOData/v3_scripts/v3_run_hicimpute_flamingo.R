#!/usr/bin/env Rscript
# Run HiCImpute MCMCImpute on one FLAMINGO v3 dataset and collect outputs.
#
# Reads binary files produced by v3_prepare_hicimpute_flamingo.py:
#   schic.bin / expected.bin / bulk.bin  (features x cells, R col-major order)
#   feature_order.npy                    (numpy row-major -> R col-major permutation)
#
# Runs MCMCImpute, then restores numpy row-major order in the output NPZ
# so the prediction aligns with the GT in the h5ad (np.triu_indices row-major)
# used by 13_cal_FLAMINGO_Baseline_metrics.py.
#
# Outputs (under <output-root>):
#   bin/<stem>_Impute_All.bin / _Impute_SZ.bin  (R col-major, backup)
#   rds/<stem>_hicimpute_result.rds            (full RDS for debugging)
#   npz_lower_tri/<stem>_hicimpute_Impute_All_lower_tri.npz
#       (cells x features, numpy row-major, scipy sparse CSR)
#
# Usage:
#   Rscript v3_run_hicimpute_flamingo.R \
#     --dataset v3_hybrid_W0p7_500cells_level0 \
#     --input-root <dir> --output-root <dir> \
#     --niter 5000 --burnin 1000 --mc-cores 40 --seed 1234 \
#     --python /path/to/python  (for reticulate npz writing)

argval <- function(args, key, default = NULL) {
  idx <- which(args == key)
  if (length(idx) == 0L) return(default)
  if (idx == length(args)) stop(paste("missing value for", key))
  args[idx + 1L]
}

flag <- function(args, key) key %in% args

main <- function() {
  suppressPackageStartupMessages(library(HiCImpute))

  args <- commandArgs(trailingOnly = TRUE)
  dataset <- argval(args, "--dataset")
  input_root <- argval(args, "--input-root")
  output_root <- argval(args, "--output-root")
  n_beads <- as.integer(argval(args, "--n-beads", "500"))
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
  exp_file <- file.path(in_dir, "expected.bin")
  order_file <- file.path(in_dir, "feature_order.npy")
  stopifnot(file.exists(schic_file), file.exists(bulk_file),
            file.exists(exp_file), file.exists(order_file))

  # Determine n_cells from file size
  fsize <- file.size(schic_file)
  n_cells <- as.integer(fsize / (8 * n_features))
  stopifnot(fsize == 8 * n_features * n_cells)

  cat(sprintf("[hicimpute] %s: reading %d x %d (features x cells)\n",
              dataset, n_features, n_cells))
  schic <- matrix(readBin(schic_file, "double", n = n_features * n_cells,
                         endian = "little"), nrow = n_features, ncol = n_cells)
  bulk <- readBin(bulk_file, "double", n = n_features, endian = "little")
  expected <- matrix(readBin(exp_file, "double", n = n_features * n_cells,
                             endian = "little"), nrow = n_features, ncol = n_cells)

  if (any(rowSums(schic) == 0)) warning("some features have zero total across cells")
  if (sum(bulk) <= 0) stop("bulk sum must be positive")

  # Load the permutation: numpy row-major -> R col-major.
  # npy is a simple binary; we read it via Python to avoid npy parsing in R.
  # Use a temp .py file instead of -c to avoid R system2 shell-quoting issues.
  if (nchar(python_bin) == 0L) python_bin <- "python3"
  read_order_py <- tempfile(pattern = "read_order_", fileext = ".py")
  on.exit(unlink(read_order_py), add = TRUE)
  writeLines(sprintf(
    "import numpy as np\no = np.load(%s)\nprint(' '.join(map(str, o.tolist())))",
    shQuote(order_file, type = "sh")), read_order_py)
  order_str <- system2(python_bin, read_order_py, stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(order_str, "status")) && attr(order_str, "status") != 0L) {
    cat("Python stderr:\n"); writeLines(order_str, stderr()); stop("reading feature_order.npy failed")
  }
  order <- as.integer(unlist(strsplit(trimws(paste(order_str, collapse = " ")), "\\s+")))
  if (length(order) != n_features)
    stop(sprintf("feature_order length %d != n_features %d", length(order), n_features))
  # order[k] = numpy row-major index that maps to R col-major position k.
  # To restore numpy row-major from R output:  pred_numpy[order+1, ] = pred_r
  # i.e. R col-major row k -> numpy row-major row order[k].

  set.seed(seed)
  startval <- c(100, 100, 10, 8, 10, 0.1, 900, 0.2, 0, rep(8, n_cells))

  dir.create(file.path(output_root, "bin"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(output_root, "rds"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(output_root, "npz_lower_tri"), recursive = TRUE, showWarnings = FALSE)

  cat(sprintf("[hicimpute] %s: MCMCImpute n=%d niter=%d burnin=%d mc.cores=%d\n",
              dataset, n_beads, niter, burnin, mc_cores))
  t0 <- proc.time()
  result <- MCMCImpute(scHiC = schic, bulk = bulk, expected = expected,
                      startval = startval, n = n_beads,
                      mc.cores = mc_cores, cutoff = 0.5,
                      niter = niter, burnin = burnin)
  cat(sprintf("[hicimpute] %s: MCMCImpute done in %.1f min\n",
              dataset, (proc.time() - t0)["elapsed"] / 60))

  saveRDS(list(dataset_id = dataset, n_beads = n_beads, n_features = n_features,
               n_cells = n_cells, niter = niter, burnin = burnin,
               mc_cores = mc_cores, seed = seed, result = result),
          file.path(output_root, "rds", paste0(dataset, "_hicimpute_result.rds")))

  # Save raw R col-major binaries (backup)
  writeBin(as.double(result$Impute_All),
           file.path(output_root, "bin", paste0(dataset, "_Impute_All.bin")),
           size = 8L, endian = "little")
  writeBin(as.double(result$Impute_SZ),
           file.path(output_root, "bin", paste0(dataset, "_Impute_SZ.bin")),
           size = 8L, endian = "little")

  # Restore numpy row-major order and write NPZ via reticulate/Python.
  # result$Impute_All is (n_features x n_cells) in R col-major order.
  impute_all_r <- result$Impute_All  # features x cells, R col-major
  # Permute feature rows back to numpy row-major: row order[k] (R) -> row k (numpy)
  impute_all_numpy <- impute_all_r[order + 1L, , drop = FALSE]  # features x cells
  # Transpose to cells x features
  pred_cells_by_features <- t(impute_all_numpy)  # cells x features, numpy row-major

  # Write NPZ using Python (scipy.sparse.save_npz CSR).
  # Write the prediction (cells x features, numpy row-major) to a temp .bin,
  # then call a Python helper to reshape and save as scipy CSR NPZ.
  npz_path <- file.path(output_root, "npz_lower_tri",
                        paste0(dataset, "_hicimpute_Impute_All_lower_tri.npz"))
  tmp_bin <- tempfile(pattern = "pred_", fileext = ".bin")
  # as.vector(t(pred)) gives row-major order so Python reshape((n_cells, n_features))
  # (default C order) reconstructs cells x features correctly.
  writeBin(as.double(as.vector(t(pred_cells_by_features))),
           tmp_bin, size = 8L, endian = "little")
  on.exit(unlink(tmp_bin), add = TRUE)

  py_script <- sprintf(
    paste0("import numpy as np\n",
           "from scipy.sparse import coo_matrix, save_npz\n",
           "v = np.fromfile(%s, dtype=np.float64)\n",
           "v = v.reshape((%d, %d))\n",
           "save_npz(%s, coo_matrix(v).tocsr())\n"),
    shQuote(tmp_bin, type = "sh"), n_cells, n_features,
    shQuote(npz_path, type = "sh"))
  py_tmp <- tempfile(pattern = "write_npz_", fileext = ".py")
  writeLines(py_script, py_tmp)
  on.exit(unlink(py_tmp), add = TRUE)
  rc <- system2(python_bin, py_tmp, stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(rc, "status")) && attr(rc, "status") != 0L) {
    cat("NPZ write failed:\n"); writeLines(rc, stderr())
    stop("NPZ write failed")
  }
  cat(sprintf("[hicimpute] %s: wrote %s\n", dataset, npz_path))
  cat(sprintf("[hicimpute] %s: done\n", dataset))
}

main()