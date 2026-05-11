"""
run_pipeline.py — Bioengineering Xenium subset pipeline runner.

Configures the 12-puck, 3-row grid layout and calls each pipeline step.
Run from this directory:  python run_pipeline.py
"""

import sys
from pathlib import Path

# ── Pipeline scripts location ───────────────────────────────────────────────
PIPELINE_DIR = Path("/Volumes/T5 EVO/Xenium_Raredon/PyScripts")
sys.path.insert(0, str(PIPELINE_DIR))

import config

# ── Override config for this project ────────────────────────────────────────
CSV_DIR = Path(__file__).parent

config.ORIG = Path(
    "/Volumes/T5 EVO/Xenium_Raredon/RQ41335-001_Raredon-849-1/"
    "output-XETG00201__0073923__Region_1__20260402__174410"
)

config.PUCK_CSVS = [
    CSV_DIR / "csv0.csv",   # PNX0, F4 RAL OGP #00390 Sp
    CSV_DIR / "csv1.csv",   # PNX0, F4 RAL OGP #00390 A
    CSV_DIR / "csv2.csv",   # LEP-DL1_RUL #00189
    CSV_DIR / "csv3.csv",   # LEP-DL1_AL #00190
    CSV_DIR / "csv4.csv",   # LEP-DL3_RML #00198
    CSV_DIR / "csv5.csv",   # LEP-DL2_RML #00195
    CSV_DIR / "csv6.csv",   # LEP-DL2_AL #00196
    CSV_DIR / "csv7.csv",   # BDL2_RUL_SM #00175
    CSV_DIR / "csv8.csv",   # 01.19.24-BDL1_RAL #00159
    CSV_DIR / "csv9.csv",   # BDL2_LLL_SM #00174
    CSV_DIR / "csv10.csv",  # 01.19.24_BDL1_LLL #00162
    CSV_DIR / "csv11.csv",  # BDL3_LLL #00180
]

config.LAYOUT = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]

config.OUT = CSV_DIR / "Xenium_subset"

config.PUCK_GAP_UM = 500

config.MORPH_WORKERS = 0  # auto

# ── Run each step ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib

    steps = [
        "01_metadata",
        "02_cells",
        "03_cells_zarr",
        "04_cells_parquet",
        "05_morphology_main",
        "06_morphology_focus",
        "07_transcripts",
    ]

    for step_name in steps:
        print(f"\n{'='*70}")
        print(f"  RUNNING: {step_name}")
        print(f"{'='*70}\n")
        mod = importlib.import_module(step_name)
        mod.main()
        print()
