"""
04_cells_parquet.py — Subset cells.parquet, cell_boundaries.parquet,
                      nucleus_boundaries.parquet.

Adds puck_id column to cells.parquet for tracking cell origin.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM
    from puck_helpers import build_puck_map, dx_dy_for_cells, puck_info_for_cells

    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )

    for name, x_col, y_col in [
        ("cells.parquet",              "x_centroid", "y_centroid"),
        ("cell_boundaries.parquet",    "vertex_x",   "vertex_y"),
        ("nucleus_boundaries.parquet", "vertex_x",   "vertex_y"),
    ]:
        src = ORIG / name
        if not src.exists():
            print(f"  - {name} not in source, skipped")
            continue
        print(f"Reading {name} …")
        df = pd.read_parquet(src)
        df.columns = [c.strip() for c in df.columns]

        cid_col = "cell_id"
        if cid_col in df.columns:
            df[cid_col] = df[cid_col].astype(str)
        df = df[df[cid_col].isin(all_cell_ids)].copy()

        dx, dy = dx_dy_for_cells(df[cid_col].values, puck_map, bboxes)
        df[x_col] = (df[x_col].values + dx).astype(np.float32)
        df[y_col] = (df[y_col].values + dy).astype(np.float32)

        if name == "cells.parquet":
            puck_id, puck_name = puck_info_for_cells(
                df[cid_col].values, puck_map, bboxes)
            df["puck_id"] = puck_id
            df["puck_name"] = puck_name

        df.to_parquet(OUT / name, index=False)
        print(f"  Written: {name}  ({len(df):,} rows)")

    print("\n✓ 04_cells_parquet.py complete.")


if __name__ == "__main__":
    main()
