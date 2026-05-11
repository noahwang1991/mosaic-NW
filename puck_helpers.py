"""
puck_helpers.py — Shared helper for the Xenium puck-subset pipeline.

Supports grid layouts: pucks arranged in rows with configurable gaps.
  LAYOUT = [[0,1,2],[3,4,5,6],[7,8,9]]  →  3 rows (3, 4, 3 pucks)

Translation convention:
  For each row, cursor_x starts at 0.
  Within a row, pucks are placed left-to-right with PUCK_GAP_UM between them.
  Rows are stacked top-to-bottom with PUCK_GAP_UM between them.
  A global floor correction ensures all new coordinates ≥ 0.
  MARGIN_UM expands bboxes for spatial filtering only.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MARGIN_UM   = 150
PIXEL_SIZE  = 0.2125

N_CODEWORDS   = 541
N_Y_FULL      = 1952
N_X_FULL      = 852
N_Y_METRICS   = 976
N_X_METRICS   = 426
TILE_SIZES_UM = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]


def _read_puck_csv(path: Path) -> tuple[str, pd.DataFrame]:
    raw  = path.read_text(encoding="utf-8", errors="replace").splitlines()
    name = path.stem
    data_lines: list[str] = []
    for line in raw:
        clean = line.strip().strip('"')
        if clean.startswith("#"):
            m = re.match(r"^#\s*Selection name\s*:\s*(.+)", clean, re.IGNORECASE)
            if m:
                name = m.group(1).strip().strip('"').replace('""', '"')
        elif line.startswith("#"):
            pass
        else:
            data_lines.append(line)
    if not data_lines:
        raise ValueError(f"No data rows found in {path}")
    df = pd.read_csv(StringIO("\n".join(data_lines)))
    return name, df


def _cell_id_column(df: pd.DataFrame, path: Path) -> str:
    for col in df.columns:
        if col.strip().lower().replace(" ", "_") == "cell_id":
            return col
    raise ValueError(
        f"Could not find a 'Cell ID' column in {path}.\n"
        f"Columns present: {list(df.columns)}"
    )


def _resolve_layout(n_pucks: int, layout) -> list[list[int]]:
    """Convert layout spec to list-of-lists of puck indices."""
    if layout is None or layout == "row":
        return [list(range(n_pucks))]
    if isinstance(layout, list) and all(isinstance(r, list) for r in layout):
        flat = [i for row in layout for i in row]
        if sorted(flat) != list(range(n_pucks)):
            raise ValueError(
                f"LAYOUT must reference each puck index 0..{n_pucks-1} exactly once.\n"
                f"Got: {layout}"
            )
        return layout
    raise ValueError(f"Invalid LAYOUT: {layout}")


def build_puck_map(
    puck_csv_paths,
    cells_csv_gz_path,
    puck_gap_um: float = 500,
    layout=None,
    verbose: bool = True,
) -> tuple[dict[str, str], set[str], pd.DataFrame, list[dict[str, Any]]]:
    """
    Returns
    -------
    puck_map   : {cell_id_str -> puck_name}
    all_cell_ids : set of every selected cell ID string
    puck_stats : DataFrame (one row per puck)
    bboxes     : list of dicts per puck (in CSV-path order)
    """
    # ── 1. Parse puck CSVs ────────────────────────────────────────────────────
    puck_names: list[str] = []
    puck_ids:   dict[str, set[str]] = {}

    for raw_path in puck_csv_paths:
        path = Path(raw_path)
        name, df = _read_puck_csv(path)
        id_col   = _cell_id_column(df, path)
        ids      = set(df[id_col].astype(str).str.strip())
        if name in puck_ids:
            name = f"{name}__{path.stem}"
        puck_names.append(name)
        puck_ids[name] = ids
        if verbose:
            print(f"  Loaded puck '{name}': {len(ids)} cells  [{path.name}]")

    n_pucks = len(puck_names)
    grid = _resolve_layout(n_pucks, layout)

    # ── 2. Read full cells table ──────────────────────────────────────────────
    cells_full = pd.read_csv(
        cells_csv_gz_path, compression="gzip",
        dtype={"cell_id": str}, low_memory=False,
    )
    cells_full.columns = [c.strip() for c in cells_full.columns]

    all_cell_ids: set[str] = set()
    for ids in puck_ids.values():
        all_cell_ids |= ids

    missing = all_cell_ids - set(cells_full["cell_id"])
    if missing and verbose:
        print(f"\n  ⚠  {len(missing)} selected cell IDs not in cells.csv.gz")
    if verbose:
        print(f"\n  Total selected cells: {len(all_cell_ids)}")

    # ── 3. Compute per-puck extents ──────────────────────────────────────────
    puck_extents: dict[str, dict] = {}
    for name in puck_names:
        mask = cells_full["cell_id"].isin(puck_ids[name])
        sub  = cells_full[mask]
        if sub.empty:
            raise ValueError(f"Puck '{name}': no matching cells in cells.csv.gz")
        puck_extents[name] = dict(
            x_min  = float(sub["x_centroid"].min()),
            x_max  = float(sub["x_centroid"].max()),
            y_min  = float(sub["y_centroid"].min()),
            y_max  = float(sub["y_centroid"].max()),
            src_cx = float(sub["x_centroid"].mean()),
            src_cy = float(sub["y_centroid"].mean()),
            n_cells = int(mask.sum()),
        )

    # ── 4. Grid layout: compute dx/dy per puck ──────────────────────────────
    # Within-row: pucks advance along y (R's horizontal axis).
    # Between rows: rows advance along x in reverse order so that
    # row 0 gets the largest x (top of R's vertical axis).
    # In Xenium Explorer, rotate 270° CW to see the intended layout.
    bboxes_by_name: dict[str, dict[str, Any]] = {}

    row_widths = []
    for row_indices in grid:
        row_names = [puck_names[i] for i in row_indices]
        rw = max(
            puck_extents[n]["x_max"] - puck_extents[n]["x_min"]
            for n in row_names
        )
        row_widths.append(rw)
    total_x = sum(row_widths) + (len(grid) - 1) * puck_gap_um

    cursor_x = total_x
    for row_indices, rw in zip(grid, row_widths):
        cursor_x -= rw
        row_names = [puck_names[i] for i in row_indices]
        cursor_y = 0.0
        for name in row_names:
            ext = puck_extents[name]
            width  = ext["x_max"] - ext["x_min"]
            height = ext["y_max"] - ext["y_min"]

            tgt_cx = cursor_x + rw / 2.0
            tgt_cy = cursor_y + height / 2.0
            dx = tgt_cx - ext["src_cx"]
            dy = tgt_cy - ext["src_cy"]

            bboxes_by_name[name] = dict(
                puck_name   = name,
                n_cells     = ext["n_cells"],
                x_min_orig  = ext["x_min"] - MARGIN_UM,
                x_max_orig  = ext["x_max"] + MARGIN_UM,
                y_min_orig  = ext["y_min"] - MARGIN_UM,
                y_max_orig  = ext["y_max"] + MARGIN_UM,
                x_min_new   = (ext["x_min"] - MARGIN_UM) + dx,
                x_max_new   = (ext["x_max"] + MARGIN_UM) + dx,
                y_min_new   = (ext["y_min"] - MARGIN_UM) + dy,
                y_max_new   = (ext["y_max"] + MARGIN_UM) + dy,
                dx = dx, dy = dy,
            )
            cursor_y += height + puck_gap_um
        cursor_x -= puck_gap_um

    # ── 5. Floor correction ──────────────────────────────────────────────────
    x_parts, y_parts = [], []
    for name in puck_names:
        mask = cells_full["cell_id"].isin(puck_ids[name])
        sub  = cells_full[mask]
        bb   = bboxes_by_name[name]
        x_parts.append(sub["x_centroid"].values + bb["dx"])
        y_parts.append(sub["y_centroid"].values + bb["dy"])

    x_min_actual = float(np.concatenate(x_parts).min())
    y_min_actual = float(np.concatenate(y_parts).min())
    x_corr = max(0.0, -x_min_actual)
    y_corr = max(0.0, -y_min_actual)

    if x_corr > 0 or y_corr > 0:
        if verbose:
            print(f"  Floor correction: Δx=+{x_corr:.4f}  Δy=+{y_corr:.4f}")
        for bb in bboxes_by_name.values():
            for k in ("dx", "x_min_new", "x_max_new"):
                bb[k] += x_corr
            for k in ("dy", "y_min_new", "y_max_new"):
                bb[k] += y_corr

    # ── 6. Output in CSV-path order ──────────────────────────────────────────
    bboxes = [bboxes_by_name[n] for n in puck_names]

    puck_map: dict[str, str] = {}
    for name, ids in puck_ids.items():
        for cid in ids:
            puck_map[cid] = name

    stats_rows = []
    for name, bb in zip(puck_names, bboxes):
        ext = puck_extents[name]
        stats_rows.append({
            "puck_name": name, "n_cells": bb["n_cells"],
            "x_min": ext["x_min"], "x_max": ext["x_max"],
            "y_min": ext["y_min"], "y_max": ext["y_max"],
            "src_cx": ext["src_cx"], "src_cy": ext["src_cy"],
            "dx": bb["dx"], "dy": bb["dy"],
        })
    puck_stats = pd.DataFrame(stats_rows)

    if verbose:
        print(f"\n{'─'*70}")
        print("  FINAL PUCK LAYOUT")
        print(f"{'─'*70}")
        for ri, row_indices in enumerate(grid):
            names_str = ", ".join(puck_names[i] for i in row_indices)
            print(f"  Row {ri}: [{names_str}]")
        for bb in bboxes:
            print(f"\n  [{bb['puck_name']}]  n_cells={bb['n_cells']}")
            print(f"    new x: [{bb['x_min_new']:.1f}, {bb['x_max_new']:.1f}]  "
                  f"y: [{bb['y_min_new']:.1f}, {bb['y_max_new']:.1f}]")
            print(f"    dx={bb['dx']:.4f}  dy={bb['dy']:.4f}")
        print(f"{'─'*70}\n")

    return puck_map, all_cell_ids, puck_stats, bboxes


# ── Spatial-filter utilities ────────────────────────────────────────────────

def in_any_puck(x_arr, y_arr, bboxes):
    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    mask  = np.zeros(len(x_arr), dtype=bool)
    for bb in bboxes:
        mask |= (
            (x_arr >= bb["x_min_orig"]) & (x_arr <= bb["x_max_orig"]) &
            (y_arr >= bb["y_min_orig"]) & (y_arr <= bb["y_max_orig"])
        )
    return mask


def get_dx_dy(x_arr, y_arr, bboxes):
    """Assign each point to its nearest puck center when bboxes overlap."""
    x_arr  = np.asarray(x_arr, dtype=np.float64)
    y_arr  = np.asarray(y_arr, dtype=np.float64)
    dx_out = np.zeros(len(x_arr), dtype=np.float32)
    dy_out = np.zeros(len(x_arr), dtype=np.float32)
    best_dist = np.full(len(x_arr), np.inf, dtype=np.float64)
    for bb in bboxes:
        m = (
            (x_arr >= bb["x_min_orig"]) & (x_arr <= bb["x_max_orig"]) &
            (y_arr >= bb["y_min_orig"]) & (y_arr <= bb["y_max_orig"])
        )
        cx = (bb["x_min_orig"] + bb["x_max_orig"]) / 2
        cy = (bb["y_min_orig"] + bb["y_max_orig"]) / 2
        dist = np.where(m, (x_arr - cx)**2 + (y_arr - cy)**2, np.inf)
        closer = dist < best_dist
        dx_out[closer] = np.float32(bb["dx"])
        dy_out[closer] = np.float32(bb["dy"])
        best_dist[closer] = dist[closer]
    return dx_out, dy_out


def dx_dy_for_cells(cell_ids, puck_map, bboxes):
    bb_by_name = {bb["puck_name"]: bb for bb in bboxes}
    ids = list(cell_ids)
    dx  = np.zeros(len(ids), dtype=np.float32)
    dy  = np.zeros(len(ids), dtype=np.float32)
    for i, cid in enumerate(ids):
        pname = puck_map.get(cid)
        if pname is not None:
            dx[i] = np.float32(bb_by_name[pname]["dx"])
            dy[i] = np.float32(bb_by_name[pname]["dy"])
    return dx, dy


def bbox_px(bb, pixel_size=PIXEL_SIZE):
    return {
        "col_start": max(0, int(bb["x_min_orig"] / pixel_size)),
        "col_end":   max(0, int(np.ceil(bb["x_max_orig"] / pixel_size))),
        "row_start": max(0, int(bb["y_min_orig"] / pixel_size)),
        "row_end":   max(0, int(np.ceil(bb["y_max_orig"] / pixel_size))),
    }


def puck_info_for_cells(cell_ids, puck_map, bboxes):
    """Return (puck_id_array, puck_name_array) aligned to cell_ids."""
    name_to_idx = {bb["puck_name"]: i for i, bb in enumerate(bboxes)}
    ids = np.asarray(cell_ids, dtype=str)
    puck_id = np.full(len(ids), -1, dtype=np.int32)
    puck_name_arr = np.empty(len(ids), dtype=object)
    for i, cid in enumerate(ids):
        pname = puck_map.get(cid)
        if pname is not None:
            puck_id[i] = name_to_idx[pname]
            puck_name_arr[i] = pname
    return puck_id, puck_name_arr
