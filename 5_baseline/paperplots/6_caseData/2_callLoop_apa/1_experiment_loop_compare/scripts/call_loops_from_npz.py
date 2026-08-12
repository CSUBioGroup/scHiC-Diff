#!/usr/bin/env python3
"""Call loops from an upper-triangle sparse NPZ matrix."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.ndimage import convolve
from scipy.stats import zscore

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:  # pragma: no cover - depends on local env
    multipletests = None

warnings.filterwarnings("ignore")


def infer_n_bins(n_features, triu_k=None):
    """Infer matrix dimension from upper-triangle feature count.

    If triu_k is None (default), auto-detect:
      - triu(k=0) including diagonal: n*(n+1)/2
      - triu(k=1) excluding diagonal: n*(n-1)/2
    If both are valid, prefer k=0 (includes diagonal).
    """
    if triu_k is not None:
        if triu_k == 0:
            n_bins = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins + 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=0)")
            return n_bins, 0
        elif triu_k == 1:
            n_bins = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins - 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=1)")
            return n_bins, 1
        else:
            raise ValueError(f"triu_k must be 0 or 1, got {triu_k}")

    # Auto-detect: try k=0 first, then k=1
    n_bins_k0 = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k0 * (n_bins_k0 + 1) // 2 == n_features:
        return n_bins_k0, 0
    n_bins_k1 = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k1 * (n_bins_k1 - 1) // 2 == n_features:
        return n_bins_k1, 1
    raise ValueError(f"Cannot infer a valid upper-triangle matrix size from n_features={n_features}")


def flat_to_matrix(flat, n_bins, triu_k=0):
    mat = np.zeros((n_bins, n_bins), dtype=np.float32)
    idx = np.triu_indices(n_bins, k=triu_k)
    mat[idx] = flat
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def normalize_by_distance(matrix, resolution, max_dist, cap):
    n_bins = matrix.shape[0]
    norm = np.zeros_like(matrix)
    max_diag = min(max_dist // resolution + 1, n_bins)

    for diag in range(1, max_diag):
        values = matrix.diagonal(diag).copy()
        nonzero_mask = values > 0
        nonzero = values[nonzero_mask]
        if len(nonzero) > 1:
            cutoff = np.percentile(nonzero, 99)
            nonzero = np.clip(nonzero, 0, cutoff)
            nonzero = zscore(nonzero)
            nonzero[np.isnan(nonzero)] = 0
            nonzero = np.clip(nonzero, -cap, cap)
            values[~nonzero_mask] = nonzero.min() if len(nonzero) > 0 else 0
            values[nonzero_mask] = nonzero
        row = np.arange(n_bins - diag)
        col = np.arange(diag, n_bins)
        norm[row, col] = values
    return norm


def compute_local_background(matrix, pad, gap):
    width = pad * 2 + 1
    kernel = np.ones((width, width), np.float32)
    kernel[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0
    kernel = kernel / np.sum(kernel)
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    bg = convolve(matrix, kernel, mode="mirror")
    return (matrix - bg) * mask


def process_single_cell(flat_vec, n_bins, resolution, max_dist, cap, pad, gap, triu_k=0):
    contact = flat_to_matrix(flat_vec, n_bins, triu_k=triu_k)
    norm = normalize_by_distance(contact, resolution=resolution, max_dist=max_dist, cap=cap)
    local = compute_local_background(norm, pad=pad, gap=gap)
    return norm, local


def merge_cells(data_matrix, n_bins, resolution, max_dist, cap, pad, gap, triu_k=0):
    n_cells = data_matrix.shape[0]
    print(f"Merging {n_cells} cells...")

    norm_sum = norm2_sum = local_sum = local2_sum = None
    for idx in range(n_cells):
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{n_cells}")
        norm, local = process_single_cell(
            data_matrix[idx],
            n_bins=n_bins,
            resolution=resolution,
            max_dist=max_dist,
            cap=cap,
            pad=pad,
            gap=gap,
            triu_k=triu_k,
        )
        if norm_sum is None:
            norm_sum, norm2_sum = norm.copy(), norm ** 2
            local_sum, local2_sum = local.copy(), local ** 2
        else:
            norm_sum += norm
            norm2_sum += norm ** 2
            local_sum += local
            local2_sum += local ** 2

    return (
        norm_sum / n_cells,
        norm2_sum / n_cells,
        local_sum / n_cells,
        local2_sum / n_cells,
        n_cells,
    )


def select_loop_candidates(norm_mean, resolution, min_dist, max_dist):
    loop = np.where(norm_mean > 0)
    dist_filter = np.logical_and(
        (loop[1] - loop[0]) > (min_dist / resolution),
        (loop[1] - loop[0]) < (max_dist / resolution),
    )
    return loop[0][dist_filter], loop[1][dist_filter]


def paired_t_test(mean, mean_sq, loop, n_cells):
    delta = mean[loop]
    total = delta * n_cells
    total_sq = mean_sq[loop] * n_cells

    sed = np.sqrt((total_sq - total ** 2 / n_cells) / (n_cells - 1) / n_cells)
    sed[sed == 0] = np.nan
    t_score = delta / sed
    p_value = stats.t.sf(t_score, n_cells - 1)
    effect_size = 2 * t_score / np.sqrt(2 * n_cells)
    return p_value, delta, effect_size


def loop_background(norm_mean, loop, pad, gap):
    width = pad * 2 + 1

    def scan_kernel(kernel):
        kernel_score = convolve(norm_mean, kernel, mode="mirror") * (norm_mean > 0)
        return kernel_score[loop]

    kernel_bl = np.zeros((width, width), np.float32)
    kernel_bl[-pad:, :(pad - gap)] = 1
    kernel_bl[-(pad - gap):, :pad] = 1
    kernel_bl = kernel_bl / np.sum(kernel_bl)

    kernel_donut = np.ones((width, width), np.float32)
    kernel_donut[pad, :] = 0
    kernel_donut[:, pad] = 0
    kernel_donut[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0
    kernel_donut = kernel_donut / np.sum(kernel_donut)

    kernel_h = np.ones((3, width), np.float32)
    kernel_h[:, (pad - gap):(pad + gap + 1)] = 0
    kernel_h = kernel_h / np.sum(kernel_h)

    kernel_v = np.ones((width, 3), np.float32)
    kernel_v[(pad - gap):(pad + gap + 1), :] = 0
    kernel_v = kernel_v / np.sum(kernel_v)

    return (
        scan_kernel(kernel_bl),
        scan_kernel(kernel_donut),
        scan_kernel(kernel_h),
        scan_kernel(kernel_v),
    )


def find_summit(loop_df, resolution, dist_thres):
    if loop_df.empty:
        return loop_df

    dist_bins = dist_thres // resolution
    coords = loop_df[["x", "y"]].values
    order = np.argsort(coords[:, 0])
    neighbors = {idx: [] for idx in range(len(order))}

    for left in range(len(order) - 1):
        current = coords[order[left]]
        for right in range(left + 1, len(order)):
            if coords[order[right], 0] - current[0] > dist_bins:
                break
            if np.abs(current[1] - coords[order[right], 1]) <= dist_bins:
                neighbors[order[left]].append(order[right])
                neighbors[order[right]].append(order[left])

    score = loop_df["E"].values
    visited = np.zeros(len(score))
    summit = []
    total = len(score)
    heap = loop_df.assign(heap_score=-loop_df["E"]).reset_index().reset_index()[["heap_score", "level_0"]].values.tolist()
    from heapq import heapify, heappop

    heapify(heap)
    while total > 0:
        top = int(heappop(heap)[1])
        while visited[top]:
            if not heap:
                break
            top = int(heappop(heap)[1])
        if visited[top]:
            break

        queue = [top]
        visited[top] = 1
        total -= 1
        cursor = 0
        local_seen = np.zeros(len(score))
        while cursor < len(queue):
            for neighbor in neighbors[queue[cursor]]:
                if not local_seen[neighbor] and score[neighbor] < score[queue[cursor]]:
                    if not visited[neighbor]:
                        visited[neighbor] = 1
                        total -= 1
                    local_seen[neighbor] = 1
                    queue.append(neighbor)
            cursor += 1
        summit.append([queue[0], len(queue)])

    summit = np.array(summit)
    output = loop_df.iloc[summit[:, 0]].copy()
    output["size"] = summit[:, 1]
    return output


def call_loops(
    data_matrix,
    output_prefix,
    resolution,
    n_bins,
    min_dist,
    max_dist,
    cap,
    pad,
    gap,
    fdr_thres,
    dist_thres,
    size_thres,
    triu_k=0,
    thres_bl=1.33,
    thres_donut=1.33,
    thres_h=1.2,
    thres_v=1.2,
):
    if multipletests is None:
        raise ImportError("statsmodels is required for FDR correction. Please install statsmodels in the runtime environment.")

    print("Step 1: merge cells")
    norm_mean, norm2_mean, local_mean, local2_mean, n_cells = merge_cells(
        data_matrix,
        n_bins=n_bins,
        resolution=resolution,
        max_dist=max_dist,
        cap=cap,
        pad=pad,
        gap=gap,
        triu_k=triu_k,
    )

    print("Step 2: select loop candidates")
    loop = select_loop_candidates(norm_mean, resolution=resolution, min_dist=min_dist, max_dist=max_dist)

    print("Step 3: paired t-test")
    local_pval, loop_t, local_d = paired_t_test(local_mean, local2_mean, loop, n_cells)
    global_pval, loop_e, global_d = paired_t_test(norm_mean, norm2_mean, loop, n_cells)

    print("Step 4: local background")
    loop_bl, loop_donut, loop_h, loop_v = loop_background(norm_mean, loop, pad=pad, gap=gap)

    print("Step 5: build result table")
    data = pd.DataFrame(
        {
            "x": loop[0],
            "y": loop[1],
            "distance": (loop[1] - loop[0]) * resolution,
            "local_pval": local_pval,
            "local_cohen_d": local_d,
            "global_pval": global_pval,
            "global_cohen_d": global_d,
            "E": loop_e,
            "T": loop_t,
            "E_bl": loop_bl,
            "E_donut": loop_donut,
            "E_h": loop_h,
            "E_v": loop_v,
        }
    )

    data["bkfilter"] = (
        ((data["E"] / data["E_bl"] > thres_bl) | (data["E_bl"] < 0))
        & ((data["E"] / data["E_donut"] > thres_donut) | (data["E_donut"] < 0))
        & ((data["E"] / data["E_h"] > thres_h) | (data["E_h"] < 0))
        & ((data["E"] / data["E_v"] > thres_v) | (data["E_v"] < 0))
    )
    data["x1"] = data["x"].astype(int) * resolution
    data["y1"] = data["y"].astype(int) * resolution
    data["x2"] = data["x1"] + resolution
    data["y2"] = data["y1"] + resolution

    print("Step 6: FDR correction")
    data.dropna(subset=["local_pval", "global_pval"], how="any", inplace=True)
    if not data.empty:
        local_qs = []
        global_qs = []
        for dist in data["distance"].unique():
            subset = data.loc[data["distance"] == dist, ["local_pval", "global_pval"]]
            _, local_q, *_ = multipletests(subset["local_pval"], method="fdr_bh")
            _, global_q, *_ = multipletests(subset["global_pval"], method="fdr_bh")
            local_qs.append(pd.Series(local_q, index=subset.index))
            global_qs.append(pd.Series(global_q, index=subset.index))
        data["local_qval"] = pd.concat(local_qs).sort_index()
        data["global_qval"] = pd.concat(global_qs).sort_index()
    else:
        data["local_qval"] = []
        data["global_qval"] = []

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(f"{output_prefix}.totalloop_info.csv", index=False)

    print("Step 7: filter loop set")
    filtered = data[
        data["bkfilter"] & (data["local_qval"] < fdr_thres) & (data["global_qval"] < fdr_thres)
    ].copy()
    filtered.to_csv(f"{output_prefix}.loop_info.csv", index=False)

    bedpe_cols = ["x1", "x2", "y1", "y2", "E"]
    if not filtered.empty:
        filtered.sort_values(by=["x1", "y1"])[bedpe_cols].to_csv(
            f"{output_prefix}.loop.bedpe",
            sep="\t",
            index=False,
            header=False,
        )
        summit = find_summit(filtered, resolution=resolution, dist_thres=dist_thres)
        summit = summit[summit["size"] >= size_thres]
        summit.sort_values(by=["x1", "y1"])[bedpe_cols + ["size"]].to_csv(
            f"{output_prefix}.loop_summit.bedpe",
            sep="\t",
            index=False,
            header=False,
        )
    else:
        Path(f"{output_prefix}.loop.bedpe").write_text("")
        Path(f"{output_prefix}.loop_summit.bedpe").write_text("")

    print(f"Finished: {filtered.shape[0]} loops")
    return data, filtered


def load_sparse_matrix(npz_path):
    payload = np.load(npz_path, allow_pickle=True)
    matrix = sparse.csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=tuple(payload["shape"]),
    )
    return matrix.toarray()


def choose_indices(total_cells, n_cells=None, seed=42, selected_indices_file=None):
    if selected_indices_file:
        indices = np.load(selected_indices_file)
        if indices.ndim != 1:
            raise ValueError(f"Selected indices file must be a 1D array: {selected_indices_file}")
        return indices.astype(int)
    if n_cells is None or n_cells >= total_cells:
        return np.arange(total_cells, dtype=int)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(total_cells, n_cells, replace=False).astype(int))


def run_case(
    input_npz,
    output_dir,
    cell_type,
    n_cells=None,
    seed=42,
    selected_indices_file=None,
    resolution=20000,
    n_bins=None,
    triu_k=None,
    min_dist=60000,
    max_dist=2000000,
    cap=5,
    pad=5,
    gap=2,
    fdr=0.05,
    dist_thres=40000,
    size_thres=1,
):
    data_matrix = load_sparse_matrix(input_npz)
    total_cells, n_features = data_matrix.shape
    if n_bins is not None and triu_k is not None:
        pass  # both specified, use as-is
    elif n_bins is not None and triu_k is None:
        # n_bins specified, infer triu_k from n_features and n_bins
        if n_bins * (n_bins + 1) // 2 == n_features:
            triu_k = 0
        elif n_bins * (n_bins - 1) // 2 == n_features:
            triu_k = 1
        else:
            raise ValueError(f"n_bins={n_bins} with n_features={n_features} matches neither triu(k=0) nor triu(k=1)")
    elif n_bins is None and triu_k is not None:
        n_bins, _ = infer_n_bins(n_features, triu_k=triu_k)
    else:
        # both None: auto-detect (prefers k=0 for ambiguous cases)
        n_bins, triu_k = infer_n_bins(n_features)
    indices = choose_indices(
        total_cells=total_cells,
        n_cells=n_cells,
        seed=seed,
        selected_indices_file=selected_indices_file,
    )
    selected = data_matrix[indices]

    output_subdir = Path(output_dir) / f"{cell_type}_{selected.shape[0]}cells"
    output_subdir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_subdir / "selected_cells.npz", sparse.csr_matrix(selected))
    np.save(output_subdir / "selected_indices.npy", indices)

    call_loops(
        selected,
        output_prefix=output_subdir / "loops",
        resolution=resolution,
        n_bins=n_bins,
        min_dist=min_dist,
        max_dist=max_dist,
        cap=cap,
        pad=pad,
        gap=gap,
        fdr_thres=fdr,
        dist_thres=dist_thres,
        size_thres=size_thres,
        triu_k=triu_k,
    )
    return output_subdir


def parse_args():
    parser = argparse.ArgumentParser(description="Call loops from a sparse upper-triangle NPZ matrix")
    parser.add_argument("--input-npz", required=True, help="Input sparse NPZ with shape (n_cells, n_features)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cell-type", required=True, help="Cell type prefix, e.g. earlyNeurons")
    parser.add_argument("--n-cells", type=int, default=None, help="Number of cells to use")
    parser.add_argument("--selected-indices-file", default=None, help="Optional .npy file with shared subset indices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, default=20000, help="Bin size in bp")
    parser.add_argument("--n-bins", type=int, default=None, help="Number of bins in the local matrix")
    parser.add_argument("--triu-k", type=int, default=None, choices=[0, 1],
                        help="Upper-triangle convention: 0=include diagonal (n*(n+1)/2), 1=exclude diagonal (n*(n-1)/2). Auto-detect if not specified.")
    parser.add_argument("--min-dist", type=int, default=60000, help="Minimum loop distance in bp")
    parser.add_argument("--max-dist", type=int, default=2000000, help="Maximum loop distance in bp")
    parser.add_argument("--cap", type=float, default=5.0, help="Z-score cap during distance normalization")
    parser.add_argument("--pad", type=int, default=5, help="Local background padding")
    parser.add_argument("--gap", type=int, default=2, help="Local background gap")
    parser.add_argument("--fdr", type=float, default=0.05, help="Loop FDR threshold")
    parser.add_argument("--dist-thres", type=int, default=40000, help="Summit merge distance in bp")
    parser.add_argument("--size-thres", type=int, default=1, help="Minimum summit cluster size")
    return parser.parse_args()


def main():
    args = parse_args()
    run_case(
        input_npz=args.input_npz,
        output_dir=args.output_dir,
        cell_type=args.cell_type,
        n_cells=args.n_cells,
        seed=args.seed,
        selected_indices_file=args.selected_indices_file,
        resolution=args.resolution,
        n_bins=args.n_bins,
        triu_k=args.triu_k,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        cap=args.cap,
        pad=args.pad,
        gap=args.gap,
        fdr=args.fdr,
        dist_thres=args.dist_thres,
        size_thres=args.size_thres,
    )


if __name__ == "__main__":
    main()
