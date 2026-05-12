"""
02_cells.py — Subset cells.csv.gz, cell_boundaries.csv.gz,
              nucleus_boundaries.csv.gz.

Adds puck_id column to cells.csv.gz for tracking cell origin.
Produces keep_idx and cell_id list used by downstream scripts (03, 04).
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

    # ── cells.csv.gz ────────────────────────────────────────────────────────
    print("Reading cells.csv.gz …")
    df = pd.read_csv(ORIG / "cells.csv.gz", compression="gzip",
                     dtype={"cell_id": str}, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    mask = df["cell_id"].isin(all_cell_ids)
    keep_idx = np.where(mask)[0]
    df = df.iloc[keep_idx].copy()
    print(f"  Subset: {len(df):,} cells")

    dx, dy = dx_dy_for_cells(df["cell_id"].values, puck_map, bboxes)
    df["x_centroid"] = (df["x_centroid"].values + dx).astype(np.float32)
    df["y_centroid"] = (df["y_centroid"].values + dy).astype(np.float32)

    puck_id, puck_name = puck_info_for_cells(df["cell_id"].values, puck_map, bboxes)
    df["puck_id"] = puck_id
    df["puck_name"] = puck_name

    np.save(OUT / ".keep_idx.npy", keep_idx)
    (OUT / ".cell_ids.txt").write_text("\n".join(df["cell_id"].values))

    df.to_csv(OUT / "cells.csv.gz", index=False, compression="gzip")
    print(f"  Written: cells.csv.gz")

    # ── cell_boundaries.csv.gz ──────────────────────────────────────────────
    print("Reading cell_boundaries.csv.gz …")
    cb = pd.read_csv(ORIG / "cell_boundaries.csv.gz", compression="gzip",
                     dtype={"cell_id": str}, low_memory=False)
    cb.columns = [c.strip() for c in cb.columns]

    cb = cb[cb["cell_id"].isin(all_cell_ids)].copy()
    dx_cb, dy_cb = dx_dy_for_cells(cb["cell_id"].values, puck_map, bboxes)
    cb["vertex_x"] = (cb["vertex_x"].values + dx_cb).astype(np.float32)
    cb["vertex_y"] = (cb["vertex_y"].values + dy_cb).astype(np.float32)
    cb.to_csv(OUT / "cell_boundaries.csv.gz", index=False, compression="gzip")
    print(f"  Written: cell_boundaries.csv.gz  ({len(cb):,} rows)")

    # ── nucleus_boundaries.csv.gz ───────────────────────────────────────────
    print("Reading nucleus_boundaries.csv.gz …")
    nb = pd.read_csv(ORIG / "nucleus_boundaries.csv.gz", compression="gzip",
                     dtype={"cell_id": str}, low_memory=False)
    nb.columns = [c.strip() for c in nb.columns]

    nb = nb[nb["cell_id"].isin(all_cell_ids)].copy()
    dx_nb, dy_nb = dx_dy_for_cells(nb["cell_id"].values, puck_map, bboxes)
    nb["vertex_x"] = (nb["vertex_x"].values + dx_nb).astype(np.float32)
    nb["vertex_y"] = (nb["vertex_y"].values + dy_nb).astype(np.float32)
    nb.to_csv(OUT / "nucleus_boundaries.csv.gz", index=False, compression="gzip")
    print(f"  Written: nucleus_boundaries.csv.gz  ({len(nb):,} rows)")

    print("\n✓ 02_cells.py complete.")


if __name__ == "__main__":
    main()
