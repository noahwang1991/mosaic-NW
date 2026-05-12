"""
03_cells_zarr.py — Subset cells.zarr.zip, cell_feature_matrix.zarr.zip,
                   cell_feature_matrix.h5, and cell_feature_matrix/ (MEX).

Matches original format:
  cells.zarr: lz4 for cell_id/cell_summary/polygon arrays, zstd for mask images
  cell_feature_matrix.zarr: zstd throughout

Requires .keep_idx.npy and .cell_ids.txt from 02_cells.py.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import zarr
    from numcodecs import Blosc
except ImportError:
    sys.exit("pip install zarr numcodecs")


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM
    from puck_helpers import build_puck_map, dx_dy_for_cells, PIXEL_SIZE, bbox_px

    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )
    N_SUBSET = len(all_cell_ids)
    keep_idx = np.load(OUT / ".keep_idx.npy")
    cell_ids = (OUT / ".cell_ids.txt").read_text().splitlines()
    dx_per_cell, dy_per_cell = dx_dy_for_cells(cell_ids, puck_map, bboxes)
    dx64 = dx_per_cell.astype(np.float64)
    dy64 = dy_per_cell.astype(np.float64)

    COMP_LZ4  = Blosc(cname="lz4",  clevel=5, shuffle=Blosc.SHUFFLE)
    COMP_ZSTD = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    KW_LZ4    = dict(overwrite=True, compressor=COMP_LZ4)
    KW_ZSTD   = dict(overwrite=True, compressor=COMP_ZSTD)

    print(f"Subset: {N_SUBSET:,} cells")

    # ─────────────────────────────────────────────────────────────────────────
    # A.  cells.zarr.zip
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── A. cells.zarr.zip ────────────────────────────────────────────────")

    src_store = zarr.ZipStore(str(ORIG / "cells.zarr.zip"), mode="r")
    src = zarr.open(src_store, mode="r")
    N_ORIG = int(src["cell_id"].shape[0])

    orig_to_new = np.full(N_ORIG, -1, dtype=np.int64)
    for new_pos, orig_pos in enumerate(keep_idx):
        orig_to_new[int(orig_pos)] = new_pos

    dst_path = OUT / "cells.zarr.zip"
    if dst_path.exists():
        dst_path.unlink()
    dst_store = zarr.ZipStore(str(dst_path), mode="w")
    dst = zarr.open(dst_store, mode="w")

    root_attrs = dict(src.attrs)
    root_attrs["number_cells"] = N_SUBSET
    dst.attrs.update(root_attrs)

    cid_sub = src["cell_id"][:][keep_idx]
    dst.create_dataset("cell_id", data=cid_sub, dtype=np.uint32,
                       chunks=cid_sub.shape, **KW_LZ4)

    cs_sub = src["cell_summary"][:][keep_idx].copy()
    cs_sub[:, 0] += dx64
    cs_sub[:, 1] += dy64
    if cs_sub.shape[1] >= 5:
        cs_sub[:, 3] += dx64
        cs_sub[:, 4] += dy64
    dst.create_dataset("cell_summary", data=cs_sub, dtype=np.float64,
                       chunks=cs_sub.shape, **KW_LZ4)
    print(f"  cell_id {cid_sub.shape}  cell_summary {cs_sub.shape}")

    masks_grp = dst.require_group("masks")
    if "masks" in src and "homogeneous_transform" in src["masks"]:
        ht = src["masks"]["homogeneous_transform"][:]
        ht_new = ht.copy()
        masks_grp.create_dataset("homogeneous_transform", data=ht_new,
                                 dtype=ht.dtype, chunks=ht.shape, **KW_LZ4)

    bb_name_to_idx = {bb["puck_name"]: i for i, bb in enumerate(bboxes)}
    orig_idx_to_puck = np.full(N_ORIG, -1, dtype=np.int32)
    for new_pos, orig_pos in enumerate(keep_idx):
        pname = puck_map.get(cell_ids[new_pos])
        if pname is not None:
            orig_idx_to_puck[int(orig_pos)] = bb_name_to_idx[pname]

    for mask_key in ["0", "1"]:
        if "masks" in src and mask_key in src["masks"]:
            print(f"  Building masks/{mask_key} …", flush=True)
            src_mask = src["masks"][mask_key]
            orig_H, orig_W = src_mask.shape

            max_new_x = max(bb["x_max_new"] for bb in bboxes)
            max_new_y = max(bb["y_max_new"] for bb in bboxes)
            canvas_W = int(np.ceil(max_new_x / PIXEL_SIZE))
            canvas_H = int(np.ceil(max_new_y / PIXEL_SIZE))

            canvas = np.zeros((canvas_H, canvas_W), dtype=np.uint32)

            for pi, bb in enumerate(bboxes):
                px = bbox_px(bb)
                r0 = px["row_start"]; r1 = min(px["row_end"], orig_H)
                c0 = px["col_start"]; c1 = min(px["col_end"], orig_W)
                if r1 <= r0 or c1 <= c0:
                    continue

                chunk = src_mask[r0:r1, c0:c1]

                paste_c = c0 + round(bb["dx"] / PIXEL_SIZE)
                paste_r = r0 + round(bb["dy"] / PIXEL_SIZE)
                skip_c = max(0, -paste_c); skip_r = max(0, -paste_r)
                dst_c = max(0, paste_c);   dst_r = max(0, paste_r)

                crop = chunk[skip_r:, skip_c:]
                hh = min(crop.shape[0], canvas_H - dst_r)
                ww = min(crop.shape[1], canvas_W - dst_c)
                if hh <= 0 or ww <= 0:
                    continue

                tile = crop[:hh, :ww].copy()
                nonzero = tile > 0
                remapped = np.zeros_like(tile)
                if nonzero.any():
                    orig_ids = tile[nonzero].astype(np.int64) - 1
                    valid = (orig_ids >= 0) & (orig_ids < N_ORIG)
                    belongs = np.zeros(len(orig_ids), dtype=bool)
                    belongs[valid] = orig_idx_to_puck[orig_ids[valid]] == pi
                    new_ids = np.zeros_like(orig_ids)
                    new_ids[valid & belongs] = orig_to_new[orig_ids[valid & belongs]]
                    new_ids[~(valid & belongs)] = -1
                    keep = new_ids >= 0
                    result = np.zeros(len(orig_ids), dtype=np.uint32)
                    result[keep] = (new_ids[keep] + 1).astype(np.uint32)
                    remapped[nonzero] = result
                canvas[dst_r:dst_r+hh, dst_c:dst_c+ww] = remapped

            orig_chunks = src_mask.chunks
            chunk_h = min(orig_chunks[0], canvas_H)
            chunk_w = min(orig_chunks[1], canvas_W)
            masks_grp.create_dataset(mask_key, data=canvas, dtype=np.uint32,
                                     chunks=(chunk_h, chunk_w), **KW_ZSTD)
            nnz = int(np.count_nonzero(canvas))
            print(f"    masks/{mask_key}: {canvas.shape}  nnz={nnz:,}")
            del canvas

    ps_src = src["polygon_sets"]
    ps_dst = dst.require_group("polygon_sets")

    for ps_key in sorted(ps_src.keys()):
        ps_in  = ps_src[ps_key]
        ps_out = ps_dst.require_group(ps_key)
        orig_ci = ps_in["cell_index"][:]

        if ps_key == "0":
            ci_clamped  = np.minimum(orig_ci.astype(np.int64), N_ORIG - 1)
            new_ci_vals = orig_to_new[ci_clamped]
            keep_mask   = new_ci_vals >= 0
            sub_ci      = new_ci_vals[keep_mask].astype(np.uint32)
            dx_ps = dx64[sub_ci][:, np.newaxis]
            dy_ps = dy64[sub_ci][:, np.newaxis]

            ps_out.create_dataset("cell_index", data=sub_ci,
                                  dtype=np.uint32, chunks=sub_ci.shape, **KW_LZ4)
            for arr_key in sorted(ps_in.keys()):
                if arr_key == "cell_index":
                    continue
                arr = ps_in[arr_key][:][keep_mask]
                if arr_key == "vertices":
                    arr = arr.copy().astype(np.float32)
                    arr[:, 0::2] += dx_ps.astype(np.float32)
                    arr[:, 1::2] += dy_ps.astype(np.float32)
                ps_out.create_dataset(arr_key, data=arr, dtype=arr.dtype,
                                      chunks=arr.shape, **KW_LZ4)
            print(f"  polygon_sets/0 (nucleus): {len(orig_ci)} → {int(keep_mask.sum())}")
        else:
            ci_new = np.arange(N_SUBSET, dtype=np.uint32)
            ps_out.create_dataset("cell_index", data=ci_new,
                                  dtype=np.uint32, chunks=ci_new.shape, **KW_LZ4)
            for arr_key in sorted(ps_in.keys()):
                if arr_key == "cell_index":
                    continue
                arr = ps_in[arr_key][:][keep_idx]
                if arr_key == "vertices":
                    arr = arr.copy().astype(np.float32)
                    arr[:, 0::2] += dx64[:, np.newaxis].astype(np.float32)
                    arr[:, 1::2] += dy64[:, np.newaxis].astype(np.float32)
                ps_out.create_dataset(arr_key, data=arr, dtype=arr.dtype,
                                      chunks=arr.shape, **KW_LZ4)
            print(f"  polygon_sets/{ps_key} (cell): {len(orig_ci)} → {N_SUBSET}")

    src_store.close()
    dst_store.close()
    print(f"  Written: cells.zarr.zip  ({dst_path.stat().st_size/1e6:.1f} MB)")

    # ─────────────────────────────────────────────────────────────────────────
    # B.  cell_feature_matrix.zarr.zip
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── B. cell_feature_matrix.zarr.zip ──────────────────────────────────")

    src2_store = zarr.ZipStore(str(ORIG / "cell_feature_matrix.zarr.zip"), mode="r")
    src2 = zarr.open(src2_store, mode="r")

    dst2_path = OUT / "cell_feature_matrix.zarr.zip"
    if dst2_path.exists():
        dst2_path.unlink()
    dst2_store = zarr.ZipStore(str(dst2_path), mode="w")
    dst2 = zarr.open(dst2_store, mode="w")

    cf_src = src2["cell_features"]
    cf_dst = dst2.require_group("cell_features")

    cf_attrs = dict(cf_src.attrs)
    cf_attrs["number_cells"] = N_SUBSET
    cf_dst.attrs.update(cf_attrs)

    cf_dst.create_dataset("cell_id", data=cf_src["cell_id"][:][keep_idx],
                          dtype=np.uint32, chunks=(N_SUBSET, 2), **KW_ZSTD)

    def _slice_csc_cols(src_grp, dst_grp, keep_idx, label):
        indptr_full  = src_grp["indptr"][:]
        data_full    = src_grp["data"][:]
        indices_full = src_grp["indices"][:]
        col_sizes    = np.diff(indptr_full)[keep_idx]
        new_indptr   = np.zeros(len(keep_idx) + 1, dtype=np.uint32)
        new_indptr[1:] = np.cumsum(col_sizes)
        parts_d, parts_i = [], []
        for oc in keep_idx:
            s, e = int(indptr_full[oc]), int(indptr_full[oc + 1])
            parts_d.append(data_full[s:e])
            parts_i.append(indices_full[s:e])
        new_data    = np.concatenate(parts_d) if parts_d else np.array([], dtype=data_full.dtype)
        new_indices = np.concatenate(parts_i) if parts_i else np.array([], dtype=indices_full.dtype)
        dst_grp.create_dataset("indptr",  data=new_indptr,  dtype=np.uint32, **KW_ZSTD)
        dst_grp.create_dataset("data",    data=new_data,    dtype=data_full.dtype,    **KW_ZSTD)
        dst_grp.create_dataset("indices", data=new_indices, dtype=indices_full.dtype, **KW_ZSTD)
        print(f"  {label}: nnz={len(new_data):,}")

    _slice_csc_cols(cf_src["csc"], cf_dst.require_group("csc"), keep_idx, "CSC")

    if all(k in cf_src for k in ("data", "indices", "indptr")):
        ip  = cf_src["indptr"][:]
        dat = cf_src["data"][:]
        idx = cf_src["indices"][:]
        n_outer = len(ip) - 1
        old_to_new = np.full(int(keep_idx.max()) + 1, -1, dtype=np.int64)
        for new_p, old_p in enumerate(keep_idx):
            old_to_new[old_p] = new_p
        new_ip = np.zeros(n_outer + 1, dtype=np.uint32)
        pd_list, pi_list = [], []
        for g in range(n_outer):
            s, e = int(ip[g]), int(ip[g + 1])
            if s == e:
                new_ip[g + 1] = new_ip[g]
                continue
            cpos = idx[s:e]
            in_r = cpos <= int(keep_idx.max())
            npos = np.full(len(cpos), -1, dtype=np.int64)
            npos[in_r] = old_to_new[cpos[in_r]]
            km = npos >= 0
            pd_list.append(dat[s:e][km])
            pi_list.append(npos[km].astype(np.uint32))
            new_ip[g + 1] = new_ip[g] + int(km.sum())
        new_dat = np.concatenate(pd_list) if pd_list else np.array([], dtype=dat.dtype)
        new_idx = np.concatenate(pi_list) if pi_list else np.array([], dtype=np.uint32)
        cf_dst.create_dataset("indptr",  data=new_ip,  dtype=np.uint32, **KW_ZSTD)
        cf_dst.create_dataset("data",    data=new_dat, dtype=dat.dtype,  **KW_ZSTD)
        cf_dst.create_dataset("indices", data=new_idx, dtype=np.uint32,  **KW_ZSTD)
        print(f"  Gene-indexed CSC: nnz={len(new_dat):,}")

    src2_store.close()
    dst2_store.close()
    print(f"  Written: cell_feature_matrix.zarr.zip  ({dst2_path.stat().st_size/1e6:.1f} MB)")

    # ─────────────────────────────────────────────────────────────────────────
    # C.  cell_feature_matrix/ (MEX directory)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── C. cell_feature_matrix/ (MEX) ────────────────────────────────────")

    import gzip
    import scipy.io as sio
    import scipy.sparse as sp

    mex_src = ORIG / "cell_feature_matrix"
    mex_dst = OUT / "cell_feature_matrix"
    mex_dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(mex_src / "features.tsv.gz", mex_dst / "features.tsv.gz")
    print("  ✓ features.tsv.gz (verbatim)")

    src_barcodes = gzip.open(mex_src / "barcodes.tsv.gz", "rt").read().splitlines()
    new_barcodes = [src_barcodes[i] for i in keep_idx]
    with gzip.open(mex_dst / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(new_barcodes) + "\n")
    print(f"  ✓ barcodes.tsv.gz ({len(new_barcodes)} barcodes)")

    mat = sio.mmread(mex_src / "matrix.mtx.gz").tocsc()
    mat_sub = mat[:, keep_idx].tocoo()
    with gzip.open(mex_dst / "matrix.mtx.gz", "wb") as f:
        sio.mmwrite(f, mat_sub)
    print(f"  ✓ matrix.mtx.gz ({mat_sub.shape}, nnz={mat_sub.nnz:,})")

    # ─────────────────────────────────────────────────────────────────────────
    # D.  cell_feature_matrix.h5
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── D. cell_feature_matrix.h5 ────────────────────────────────────────")

    try:
        import h5py
        h5_src = ORIG / "cell_feature_matrix.h5"
        h5_dst = OUT / "cell_feature_matrix.h5"
        if h5_src.exists():
            with h5py.File(h5_src, "r") as hin:
                group_name = list(hin.keys())[0]
                grp = hin[group_name]
                barcodes_h5 = grp["barcodes"][:][keep_idx]
                data_h5     = grp["data"][:]
                indices_h5  = grp["indices"][:]
                indptr_h5   = grp["indptr"][:]
                shape_h5    = grp["shape"][:]

                csc = sp.csc_matrix((data_h5, indices_h5, indptr_h5),
                                    shape=(int(shape_h5[0]), int(shape_h5[1])))
                csc_sub = csc[:, keep_idx]

                with h5py.File(h5_dst, "w") as hout:
                    g = hout.create_group(group_name)
                    g.create_dataset("barcodes", data=barcodes_h5)
                    g.create_dataset("data",     data=np.array(csc_sub.data, dtype=data_h5.dtype))
                    g.create_dataset("indices",  data=np.array(csc_sub.indices, dtype=indices_h5.dtype))
                    g.create_dataset("indptr",   data=np.array(csc_sub.indptr, dtype=indptr_h5.dtype))
                    g.create_dataset("shape",    data=np.array([csc_sub.shape[0], csc_sub.shape[1]],
                                                               dtype=shape_h5.dtype))
                    if "features" in grp:
                        hin.copy(grp["features"], g, "features")
                    for attr_key in grp.attrs:
                        g.attrs[attr_key] = grp.attrs[attr_key]
            print(f"  ✓ cell_feature_matrix.h5  ({h5_dst.stat().st_size/1e6:.1f} MB)")
        else:
            print("  - cell_feature_matrix.h5 not in source")
    except ImportError:
        print("  - h5py not available, skipping .h5")

    print("\n✓ 03_cells_zarr.py complete.")


if __name__ == "__main__":
    main()
