"""
08_analysis.py — Subset analysis.zarr.zip to keep only selected cells.

Remaps cell indices in cell_groups/ so that cluster assignments from the
original Xenium run are preserved for the subset cells.  Cluster labels
and grouping metadata (graphclust, kmeans, gene/protein) are kept intact;
clusters that lose all their members become empty but stay in the schema.

Standalone usage (paths as CLI args):
    python 08_analysis.py <ORIG_DIR> <OUT_DIR>

Pipeline usage (reads ORIG / OUT from config.py):
    Called via main() from run_pipeline.py

Requires .keep_idx.npy from 02_cells.py in OUT/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import zarr
except ImportError:
    sys.exit("pip install zarr")


def main(orig: Path | None = None, out: Path | None = None):
    if orig is None or out is None:
        from config import ORIG, OUT
        orig = orig or ORIG
        out = out or OUT

    src_path = orig / "analysis.zarr.zip"
    dst_path = out / "analysis.zarr.zip"
    keep_idx_path = out / ".keep_idx.npy"

    print(f"ORIG: {orig}")
    print(f"OUT:  {out}")

    if not src_path.exists():
        sys.exit(f"Source not found: {src_path}")
    if not keep_idx_path.exists():
        sys.exit(f"keep_idx not found: {keep_idx_path}\nRun 02_cells.py first.")

    # ── Load keep_idx → build orig_to_new mapping ───────────────────────────
    keep_idx = np.load(keep_idx_path)          # new_pos → orig_pos
    n_new = len(keep_idx)
    n_orig_max = int(keep_idx.max()) + 1

    orig_to_new = np.full(n_orig_max, -1, dtype=np.int64)
    for new_pos, orig_pos in enumerate(keep_idx):
        orig_to_new[int(orig_pos)] = new_pos

    print(f"Subset cells: {n_new:,}  (orig max index: {n_orig_max - 1:,})")

    # ── Open source zarr ────────────────────────────────────────────────────
    src = zarr.open(str(src_path), mode="r")

    if "cell_groups" not in src:
        sys.exit("No cell_groups/ found in analysis.zarr.zip")

    cg = src["cell_groups"]
    n_groupings = int(cg.attrs.get("number_groupings", 0))
    grouping_names = list(cg.attrs.get("grouping_names", []))
    group_names = list(cg.attrs.get("group_names", []))

    print(f"Groupings: {n_groupings}")
    for i, gn in enumerate(grouping_names):
        n_clusters = len(group_names[i]) if i < len(group_names) else "?"
        print(f"  [{i:2d}] {gn}  ({n_clusters} clusters)")

    # ── Build remapped groupings ────────────────────────────────────────────
    remapped = []  # list of (new_indices, new_indptr) per grouping
    total_kept = 0
    total_orig = 0

    for gi in range(n_groupings):
        indices = np.array(cg[f"{gi}/indices"], dtype=np.int64)
        indptr  = np.array(cg[f"{gi}/indptr"], dtype=np.int64)
        n_clusters = len(indptr) - 1
        total_orig += len(indices)

        new_indptr = np.zeros(n_clusters + 1, dtype=np.uint32)
        new_indices_parts = []

        for c in range(n_clusters):
            orig_ids = indices[indptr[c]:indptr[c + 1]]
            # Filter to cells that exist in subset and are within mapping range
            valid = orig_ids < n_orig_max
            mapped = orig_to_new[orig_ids[valid]]
            kept = mapped[mapped >= 0]
            kept_sorted = np.sort(kept).astype(np.uint32)
            new_indices_parts.append(kept_sorted)
            new_indptr[c + 1] = new_indptr[c] + len(kept_sorted)

        new_indices = np.concatenate(new_indices_parts) if new_indices_parts else np.array([], dtype=np.uint32)
        total_kept += len(new_indices)
        remapped.append((new_indices, new_indptr))

        gname = grouping_names[gi] if gi < len(grouping_names) else f"grouping_{gi}"
        empty_clusters = sum(1 for c in range(n_clusters)
                           if new_indptr[c + 1] == new_indptr[c])
        print(f"  [{gi:2d}] {gname}: {len(new_indices):,}/{len(indices):,} cells kept"
              f"  ({empty_clusters} empty clusters)")

    print(f"\nTotal cells in groupings: {total_kept:,} / {total_orig:,}")

    # ── Write new analysis.zarr.zip ─────────────────────────────────────────
    if dst_path.exists():
        dst_path.unlink()

    dst = zarr.open(str(dst_path), mode="w")

    # Copy cell_groups attrs
    dst_cg = dst.require_group("cell_groups")
    dst_cg.attrs["number_groupings"] = n_groupings
    dst_cg.attrs["grouping_names"] = grouping_names
    dst_cg.attrs["group_names"] = group_names

    # Write remapped arrays
    for gi, (new_indices, new_indptr) in enumerate(remapped):
        g = dst_cg.require_group(str(gi))
        g.create_dataset("indices", data=new_indices, dtype="uint32",
                         chunks=min(len(new_indices), 65536) or 1)
        g.create_dataset("indptr", data=new_indptr, dtype="uint32",
                         chunks=min(len(new_indptr), 65536) or 1)

    # Copy any other top-level groups/arrays that aren't cell_groups
    for key in src.keys():
        if key != "cell_groups":
            print(f"  Copying /{key} verbatim …")
            zarr.copy(src[key], dst, name=key)

    sz = dst_path.stat().st_size / 1e6
    print(f"\n✓ Written: {dst_path.name}  ({sz:.1f} MB)")
    print("✓ 08_analysis.py complete.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(orig=Path(sys.argv[1]), out=Path(sys.argv[2]))
    elif len(sys.argv) == 1:
        main()
    else:
        sys.exit("Usage: python 08_analysis.py [ORIG_DIR OUT_DIR]")
