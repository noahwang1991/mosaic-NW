"""
config.py — Central configuration for the Xenium puck-subset pipeline.
All downstream scripts import from here.

LAYOUT controls how pucks are arranged in the output:
  - None or "row"  → single row, left to right
  - List of lists   → grid, e.g. [[0,1,2],[3,4,5,6],[7,8,9]] = 3 rows
    Each number is a 0-based index into PUCK_CSVS.
"""

from pathlib import Path

# ── Source experiment (read-only) ─────────────────────────────────────────────
ORIG = Path(
    "/Volumes/T5 EVO/Xenium_Raredon/RQ41335-001_Raredon-849-1/"
    "output-XETG00201__0073923__Region_1__20260402__174410"
)

# ── Puck selection CSVs (order = index for LAYOUT) ───────────────────────────
PUCK_CSVS = [
    Path("/Volumes/T5 EVO/Xenium_Raredon/selected_pucks/LEP-DL1_RUL_cells_stats.csv"),
    Path("/Volumes/T5 EVO/Xenium_Raredon/selected_pucks/PNX0_F4_RAL_OGP_cells_stats.csv"),
]

# ── Layout: None → single row.  Or list-of-lists for grid. ──────────────────
# Examples:
#   LAYOUT = None                              # [puck0] [puck1]
#   LAYOUT = [[0, 1]]                          # same as above
#   LAYOUT = [[0], [1]]                        # vertical stack
#   LAYOUT = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]  # 3-4-3 grid
LAYOUT = None

# ── Output folder ────────────────────────────────────────────────────────────
OUT = Path("/Volumes/T5 EVO/Xenium_Raredon/PyScripts/brand_new/Xenium_subset")

# ── Gap between pucks (µm) ───────────────────────────────────────────────────
PUCK_GAP_UM = 500

# ── Parallel workers for morphology_focus (0 = auto) ─────────────────────────
MORPH_WORKERS = 0
