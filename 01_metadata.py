"""
01_metadata.py — Scaffold output directory and copy metadata files.

Creates OUT/, writes experiment.xenium (with updated num_cells),
copies gene_panel.json, protein_panel.json, metrics_summary.csv,
analysis.zarr.zip, analysis_summary.html, analysis/ verbatim.
Writes puck_manifest.csv recording puck_id → puck_name mapping.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM
    from puck_helpers import build_puck_map

    print("Loading puck map …")
    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )
    N_SUBSET = len(all_cell_ids)
    print(f"  Total subset cells: {N_SUBSET}")

    for d in [OUT, OUT / "morphology_focus", OUT / "cell_feature_matrix"]:
        d.mkdir(parents=True, exist_ok=True)

    # ── experiment.xenium ────────────────────────────────────────────────────
    print("Writing experiment.xenium …")
    with open(ORIG / "experiment.xenium", "r") as f:
        meta = json.load(f)

    meta["num_cells"] = N_SUBSET
    with open(OUT / "experiment.xenium", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  num_cells: {meta['num_cells']}")

    # ── Puck manifest ───────────────────────────────────────────────────────
    print("Writing puck_manifest.csv …")
    manifest_rows = []
    for i, (bb, csv_path) in enumerate(zip(bboxes, PUCK_CSVS)):
        manifest_rows.append({
            "puck_id": i,
            "puck_name": bb["puck_name"],
            "n_cells": bb["n_cells"],
            "csv_file": Path(csv_path).name,
        })
    pd.DataFrame(manifest_rows).to_csv(OUT / "puck_manifest.csv", index=False)
    for row in manifest_rows:
        print(f"  {row['puck_id']}: {row['puck_name']} ({row['n_cells']} cells)")

    # ── Verbatim files ──────────────────────────────────────────────────────
    VERBATIM = [
        "gene_panel.json",
        "protein_panel.json",
        "metrics_summary.csv",
        "index.html",
        "analysis.zarr.zip",
        "analysis_summary.html",
    ]
    print("Copying verbatim files …")
    for name in VERBATIM:
        src = ORIG / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            print(f"  ✓ {name}")
        else:
            print(f"  - {name} (not in source, skipped)")

    for dirname in ["analysis"]:
        src = ORIG / dirname
        dst = OUT / dirname
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            print(f"  ✓ {dirname}/")

    print("\n✓ 01_metadata.py complete.")


if __name__ == "__main__":
    main()
