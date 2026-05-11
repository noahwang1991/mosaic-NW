"""
07_transcripts.py — Subset transcripts.csv.gz, transcripts.parquet,
                    and transcripts.zarr.zip in one script.

zarr tile format:
  Level 0: 9 arrays (codeword_identity, gene_identity, gene_offset,
           id, location, quality_score, status, uuid, valid).
           Hi-quality (qv>=20) first, lo-quality second, both sorted by gene_identity.

  Levels 1+: 4 arrays (cluster_count, gene_identity, gene_offset, location).
           Nearby same-gene transcripts clustered spatially.
           Cluster bin size grows with level.

  gene_offset (541, 4) uint32:
    [lo_start, lo_end, hi_start, hi_end]

  Format: dimension_separator=".", order="F" (except gene_offset="C"),
          compressor=blosc/zstd, fill_value=None.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

try:
    import zarr
    from numcodecs import Blosc
    import scipy.sparse as sp
except ImportError:
    sys.exit("pip install zarr numcodecs scipy")

QV_HIGH = 20.0
COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
CLUSTER_BIN_BASE = 5.0
BIN_10 = 10.0
BIN_20 = 20.0

TILE_L0_ZATTRS = {
    "codeword_identity": {
        "column_names": ["codeword1_call", "codeword2_call"],
        "column_descriptions": ["Codeword 1 Call", "Codeword 2 Call"],
    },
    "gene_identity": {
        "column_names": ["gene_call"],
        "column_descriptions": ["Gene Call"],
    },
    "gene_offset": {
        "column_names": ["low_qscore_start", "low_qscore_end",
                         "high_qscore_start", "high_qscore_end"],
        "column_descriptions": [
            "element start index for the gene's low quality clusters",
            "element end index for the gene's low quality clusters (not inclusive)",
            "element start index for the gene's high quality clusters",
            "element end index for the gene's high quality clusters (not inclusive)",
        ],
    },
    "id": {
        "column_names": ["id", "fov_index"],
        "column_descriptions": ["Rna Identifier", "FOV Index"],
    },
    "location": {
        "column_names": ["x_position", "y_position", "z_position"],
        "column_descriptions": ["X Position", "Y Position", "Z Position"],
    },
    "quality_score": {
        "column_names": ["calibrated_codeword_score"],
        "column_descriptions": ["Calibrated Codeword Score"],
    },
    "status": {
        "column_names": ["status"],
        "column_descriptions": ["Rna Status"],
    },
    "uuid": {
        "column_names": ["uuid_w0", "uuid_w1"],
        "column_descriptions": ["Blob Unique Identifier, Word 0",
                                "Blob Unique Identifier, Word 1"],
    },
    "valid": {
        "column_names": ["valid"],
        "column_descriptions": ["Valid"],
    },
}

TILE_CLUSTER_ZATTRS = {
    "cluster_count": {
        "column_names": ["cluster_count"],
        "column_descriptions": ["Count of transcripts grouped into this cluster"],
    },
    "gene_identity": TILE_L0_ZATTRS["gene_identity"],
    "gene_offset":   TILE_L0_ZATTRS["gene_offset"],
    "location":      TILE_L0_ZATTRS["location"],
}


def _create_tile_array(grp, name, data, dtype, zattrs_map):
    order = "C" if name == "gene_offset" else "F"
    grp.create_dataset(
        name, data=data, dtype=dtype, chunks=data.shape,
        compressor=COMPRESSOR, fill_value=None, order=order,
        dimension_separator=".", overwrite=True,
    )
    if name in zattrs_map:
        grp[name].attrs.update(zattrs_map[name])


def _compute_gene_offset(gi_sorted, n_hi, n_codewords):
    go = np.zeros((n_codewords, 4), dtype=np.uint32)
    hi_gi = gi_sorted[:n_hi]
    lo_gi = gi_sorted[n_hi:]
    for g in range(n_codewords):
        hi_hits = np.where(hi_gi == g)[0]
        if len(hi_hits):
            go[g, 2] = int(hi_hits[0])
            go[g, 3] = int(hi_hits[-1]) + 1
        lo_hits = np.where(lo_gi == g)[0]
        if len(lo_hits):
            go[g, 0] = n_hi + int(lo_hits[0])
            go[g, 1] = n_hi + int(lo_hits[-1]) + 1
    return go


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM
    from puck_helpers import (build_puck_map, in_any_puck, get_dx_dy,
                               N_CODEWORDS, N_Y_FULL, N_X_FULL,
                               N_Y_METRICS, N_X_METRICS, TILE_SIZES_UM)

    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Load, filter, translate
    # ─────────────────────────────────────────────────────────────────────────
    print("Phase 1: Loading and filtering transcripts …")
    df_all = pd.read_csv(ORIG / "transcripts.csv.gz", compression="gzip",
                         dtype={"cell_id": str}, low_memory=False)
    df_all.columns = [c.strip() for c in df_all.columns]

    mask = in_any_puck(df_all["x_location"].values, df_all["y_location"].values, bboxes)
    df   = df_all[mask].copy()
    del df_all
    print(f"  Subset rows: {len(df):,}")

    dx_arr, dy_arr = get_dx_dy(df["x_location"].values, df["y_location"].values, bboxes)
    df["x_new"] = (df["x_location"].values + dx_arr).astype(np.float32)
    df["y_new"] = (df["y_location"].values + dy_arr).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Write CSV and Parquet
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPhase 2: Writing CSV and Parquet …")
    df_out = df.copy()
    df_out["x_location"] = df_out["x_new"]
    df_out["y_location"] = df_out["y_new"]
    df_out.drop(columns=["x_new", "y_new"], inplace=True)

    df_out.to_csv(OUT / "transcripts.csv.gz", index=False, compression="gzip")
    print(f"  ✓ transcripts.csv.gz  ({len(df_out):,} rows)")

    df_pq = pd.read_parquet(ORIG / "transcripts.parquet")
    df_pq.columns = [c.strip() for c in df_pq.columns]
    mask_pq = in_any_puck(df_pq["x_location"].values.astype(float),
                          df_pq["y_location"].values.astype(float), bboxes)
    df_pq = df_pq[mask_pq].copy()
    dx_pq, dy_pq = get_dx_dy(df_pq["x_location"].values.astype(float),
                              df_pq["y_location"].values.astype(float), bboxes)
    df_pq["x_location"] = (df_pq["x_location"].values + dx_pq).astype(np.float32)
    df_pq["y_location"] = (df_pq["y_location"].values + dy_pq).astype(np.float32)
    df_pq.to_parquet(OUT / "transcripts.parquet", index=False)
    print(f"  ✓ transcripts.parquet  ({len(df_pq):,} rows)")
    del df_pq, df_out

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Prepare arrays for zarr
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPhase 3: Preparing zarr arrays …")
    src_store = zarr.ZipStore(str(ORIG / "transcripts.zarr.zip"), mode="r")
    src_root  = zarr.open(src_store, mode="r")
    root_attrs = dict(src_root.attrs)

    gene_index_map = root_attrs["gene_index_map"]
    fov_names    = root_attrs.get("fov_names", [])
    fov_name_map = {n: i for i, n in enumerate(fov_names)}

    N = len(df)

    df_gene_idx = np.array(
        [gene_index_map.get(str(fn), 0) for fn in df["feature_name"].values],
        dtype=np.int32
    )
    df_cw = np.clip(df["codeword_index"].values.astype(np.int32), 0, N_CODEWORDS - 1)

    x_new = df["x_new"].values.astype(np.float32)
    y_new = df["y_new"].values.astype(np.float32)
    z_arr = df["z_location"].values.astype(np.float32) if "z_location" in df.columns \
            else np.zeros(N, np.float32)

    tid = df["transcript_id"].values
    if tid.dtype == np.float64:
        tid_lo = (tid.astype(np.int64) & 0x7FFFFFFF).astype(np.uint32)
    else:
        tid_lo = (tid.astype(np.uint64) & 0xFFFFFFFF).astype(np.uint32)
    fov_idx = np.array([fov_name_map.get(str(fn), 0)
                        for fn in df["fov_name"].values], dtype=np.uint32) \
              if "fov_name" in df.columns else np.zeros(N, np.uint32)

    qv_arr = df["qv"].values.astype(np.float32)
    status = np.where(df["cell_id"].values == "UNASSIGNED", np.uint8(1), np.uint8(0))
    valid  = np.ones(N, dtype=np.uint8)
    is_gene = df["is_gene"].values.astype(bool)

    print(f"  N={N:,}  hi_qv={int((qv_arr >= QV_HIGH).sum()):,}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Density matrices
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPhase 4: Density matrices …")

    def _build_density_csr(outer_idx, x_vals, y_vals, n_outer, n_y, n_x, bin_size):
        xb = np.clip((x_vals / bin_size).astype(np.int32), 0, n_x - 1)
        yb = np.clip((y_vals / bin_size).astype(np.int32), 0, n_y - 1)
        rows = outer_idx * n_y + yb
        mat = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.uint16), (rows, xb)),
            shape=(n_outer * n_y, n_x)
        ).tocsr()
        mat.data = np.clip(mat.data, 0, 65535).astype(np.uint16)
        return mat

    mat_cw = _build_density_csr(df_cw, x_new, y_new, N_CODEWORDS, N_Y_FULL, N_X_FULL, BIN_10)
    mat_gn = _build_density_csr(df_gene_idx, x_new, y_new, N_CODEWORDS, N_Y_FULL, N_X_FULL, BIN_10)
    print(f"  density/codeword nnz={mat_cw.nnz:,}")
    print(f"  density/gene     nnz={mat_gn.nnz:,}")

    metrics = np.zeros((N_Y_METRICS, N_X_METRICS, 4), dtype=np.float32)
    xb20 = np.clip((x_new / BIN_20).astype(np.int32), 0, N_X_METRICS - 1)
    yb20 = np.clip((y_new / BIN_20).astype(np.int32), 0, N_Y_METRICS - 1)
    np.add.at(metrics[:, :, 0], (yb20, xb20), 1.0)
    np.add.at(metrics[:, :, 1], (yb20[is_gene], xb20[is_gene]), 1.0)
    np.add.at(metrics[:, :, 2], (yb20[~is_gene], xb20[~is_gene]), 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5: Build zarr
    # ─────────────────────────────────────────────────────────────────────────
    print("\nPhase 5: Building transcripts.zarr.zip …")

    zarr_path = OUT / "transcripts.zarr.zip"
    if zarr_path.exists():
        zarr_path.unlink()
    dst_store = zarr.ZipStore(str(zarr_path), mode="w")
    dst_root  = zarr.open(dst_store, mode="w")

    updated_attrs = dict(root_attrs)
    updated_attrs["number_rnas"] = int(N)
    dst_root.attrs.update(updated_attrs)

    for key in ("codeword_category", "gene_category"):
        if key in src_root:
            arr = src_root[key][:]
            dst_root.create_dataset(key, data=arr, dtype=arr.dtype,
                                    chunks=(541, 1), compressor=COMPRESSOR, overwrite=True)

    dst_grids = dst_root.require_group("grids")

    def _write_tile_level0(grp, idxs_arr):
        if len(idxs_arr) == 0:
            return
        qv_raw = qv_arr[idxs_arr]
        hi_mask = qv_raw >= QV_HIGH
        hi_idxs = idxs_arr[hi_mask]
        lo_idxs = idxs_arr[~hi_mask]
        hi_gi = df_gene_idx[hi_idxs].astype(np.uint16)
        lo_gi = df_gene_idx[lo_idxs].astype(np.uint16)
        hi_order = np.argsort(hi_gi, kind="stable")
        lo_order = np.argsort(lo_gi, kind="stable")
        ordered = np.concatenate([hi_idxs[hi_order], lo_idxs[lo_order]])
        gi_sorted = np.concatenate([hi_gi[hi_order], lo_gi[lo_order]])
        n_hi = len(hi_idxs)
        N_t = len(ordered)
        go = _compute_gene_offset(gi_sorted, n_hi, N_CODEWORDS)
        za = TILE_L0_ZATTRS
        _create_tile_array(grp, "gene_identity", gi_sorted.reshape(-1, 1), np.uint16, za)
        _create_tile_array(grp, "gene_offset", go, np.uint32, za)
        loc = np.stack([x_new[ordered], y_new[ordered], z_arr[ordered]], axis=1)
        _create_tile_array(grp, "location", loc.astype(np.float32), np.float32, za)
        ci = np.column_stack([df_cw[ordered].astype(np.uint32),
                              np.full(N_t, 0xFFFFFFFF, dtype=np.uint32)])
        _create_tile_array(grp, "codeword_identity", ci, np.uint32, za)
        id_arr = np.column_stack([tid_lo[ordered], fov_idx[ordered]])
        _create_tile_array(grp, "id", id_arr.astype(np.uint32), np.uint32, za)
        _create_tile_array(grp, "uuid", id_arr.astype(np.uint32), np.uint32, za)
        _create_tile_array(grp, "quality_score", qv_arr[ordered].reshape(-1, 1), np.float32, za)
        _create_tile_array(grp, "status", status[ordered].reshape(-1, 1), np.uint8, za)
        _create_tile_array(grp, "valid", valid[ordered].reshape(-1, 1), np.uint8, za)

    def _write_tile_clustered(grp, idxs_arr, cluster_bin_um):
        if len(idxs_arr) == 0:
            return
        gi  = df_gene_idx[idxs_arr].astype(np.int32)
        qv  = qv_arr[idxs_arr]
        x   = x_new[idxs_arr]
        y   = y_new[idxs_arr]
        z   = z_arr[idxs_arr]
        hi  = qv >= QV_HIGH
        xb = (x / cluster_bin_um).astype(np.int32)
        yb = (y / cluster_bin_um).astype(np.int32)

        clusters_hi = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        clusters_lo = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for j in range(len(idxs_arr)):
            key = (int(gi[j]), int(xb[j]), int(yb[j]))
            c = clusters_hi[key] if hi[j] else clusters_lo[key]
            c[0] += 1; c[1] += float(x[j]); c[2] += float(y[j]); c[3] += float(z[j])

        def _flatten(clusters):
            entries = []
            for (g, _, _), (cnt, sx, sy, sz) in clusters.items():
                entries.append((g, cnt, sx / cnt, sy / cnt, sz / cnt))
            entries.sort(key=lambda e: e[0])
            return entries

        hi_entries = _flatten(clusters_hi)
        lo_entries = _flatten(clusters_lo)
        all_entries = hi_entries + lo_entries
        n_hi = len(hi_entries)
        if not all_entries:
            return
        c_gi  = np.array([e[0] for e in all_entries], dtype=np.uint16)
        c_cnt = np.array([e[1] for e in all_entries], dtype=np.uint32)
        c_loc = np.array([[e[2], e[3], e[4]] for e in all_entries], dtype=np.float32)
        go = _compute_gene_offset(c_gi, n_hi, N_CODEWORDS)
        za = TILE_CLUSTER_ZATTRS
        _create_tile_array(grp, "cluster_count", c_cnt.reshape(-1, 1), np.uint32, za)
        _create_tile_array(grp, "gene_identity", c_gi.reshape(-1, 1), np.uint16, za)
        _create_tile_array(grp, "gene_offset", go, np.uint32, za)
        _create_tile_array(grp, "location", c_loc, np.float32, za)

    # ── Iterate pyramid levels ──────────────────────────────────────────────
    all_grid_keys   = []
    all_grid_nobjs  = []
    all_grid_shapes = []
    all_npg         = []

    for level, tile_um in enumerate(TILE_SIZES_UM):
        tx = (x_new / tile_um).astype(np.int32)
        ty = (y_new / tile_um).astype(np.int32)

        tile_buffers = defaultdict(list)
        for i, (txi, tyi) in enumerate(zip(tx, ty)):
            tile_buffers[(int(txi), int(tyi))].append(i)

        dst_level = dst_root.require_group(f"grids/{level}")
        cluster_bin = CLUSTER_BIN_BASE * (2 ** (level - 1)) if level > 0 else 0

        keys_l  = []
        nobjs_l = []
        hi_max  = np.zeros(N_CODEWORDS, dtype=np.int64)
        lo_max  = np.zeros(N_CODEWORDS, dtype=np.int64)

        for (txi, tyi) in sorted(tile_buffers.keys()):
            idxs = tile_buffers[(txi, tyi)]
            tk = f"{txi},{tyi}"
            tile_grp = dst_level.require_group(tk)
            idxs_np = np.array(idxs, dtype=np.int64)

            if level == 0:
                _write_tile_level0(tile_grp, idxs_np)
                n_entries = len(idxs_np)
            else:
                _write_tile_clustered(tile_grp, idxs_np, cluster_bin)
                n_entries = tile_grp["gene_identity"].shape[0] if "gene_identity" in tile_grp else 0

            keys_l.append(tk)
            nobjs_l.append(n_entries)

            gi_t = df_gene_idx[idxs_np].astype(np.int32)
            qv_t = qv_arr[idxs_np]
            hi_tile = np.zeros(N_CODEWORDS, dtype=np.int64)
            lo_tile = np.zeros(N_CODEWORDS, dtype=np.int64)
            hi_m = qv_t >= QV_HIGH
            np.add.at(hi_tile, gi_t[hi_m & (gi_t >= 0) & (gi_t < N_CODEWORDS)], 1)
            np.add.at(lo_tile, gi_t[~hi_m & (gi_t >= 0) & (gi_t < N_CODEWORDS)], 1)
            np.maximum(hi_max, hi_tile, out=hi_max)
            np.maximum(lo_max, lo_tile, out=lo_max)

        total = sum(nobjs_l)
        extra = f"  cluster_bin={cluster_bin:.0f}µm" if level > 0 else ""
        print(f"  Level {level} ({tile_um}µm): {len(keys_l)} tiles  "
              f"{total:,} entries{extra}", flush=True)

        all_grid_keys.append(keys_l)
        all_grid_nobjs.append(nobjs_l)
        all_grid_shapes.append([{} for _ in keys_l])
        all_npg.append({"high_qscore": hi_max.tolist(), "low_qscore": lo_max.tolist()})

    # ── grids attrs ─────────────────────────────────────────────────────────
    src_grids_attrs = dict(src_root["grids"].attrs)
    cw_counts = np.zeros(N_CODEWORDS, dtype=np.int64)
    for cw, cnt in zip(*np.unique(df_cw, return_counts=True)):
        cw_counts[cw] = int(cnt)
    src_grids_attrs["codeword_to_transcript_counts"] = cw_counts.tolist()
    src_grids_attrs["grid_keys"]                        = all_grid_keys
    src_grids_attrs["grid_number_objects"]              = all_grid_nobjs
    src_grids_attrs["grid_array_shapes"]                = all_grid_shapes
    src_grids_attrs["number_objects_per_tile_per_gene"] = all_npg
    src_grids_attrs["number_levels"]                    = len(TILE_SIZES_UM)
    dst_grids.attrs.update(src_grids_attrs)

    # ── Density ─────────────────────────────────────────────────────────────
    print("Writing density matrices …")
    src_density = src_root["density"] if "density" in src_root else None

    def _write_density_grp(path, mat, src_grp):
        grp = dst_root.require_group(path)
        if src_grp is not None:
            grp.attrs.update(dict(src_grp.attrs))
        grp.create_dataset("data",    data=np.asarray(mat.data, np.uint16),
                           dtype=np.uint16, compressor=COMPRESSOR, overwrite=True)
        grp.create_dataset("indices", data=np.asarray(mat.indices, np.uint16),
                           dtype=np.uint16, compressor=COMPRESSOR, overwrite=True)
        grp.create_dataset("indptr",  data=np.asarray(mat.indptr, np.uint32),
                           dtype=np.uint32, compressor=COMPRESSOR, overwrite=True)

    _write_density_grp("density/codeword", mat_cw,
                       src_density["codeword"] if src_density and "codeword" in src_density else None)
    _write_density_grp("density/gene", mat_gn,
                       src_density["gene"] if src_density and "gene" in src_density else None)

    dst_root.create_dataset("metrics_density", data=metrics,
                            dtype=np.float32, compressor=COMPRESSOR, overwrite=True)

    src_store.close()
    dst_store.close()

    sz = zarr_path.stat().st_size / 1e6
    print(f"\n  ✓ transcripts.zarr.zip  ({sz:.0f} MB)")
    print("✓ 07_transcripts.py complete.")


if __name__ == "__main__":
    main()
