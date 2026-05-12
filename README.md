# mosaic

Subset and rearrange tissue pucks from a 10x Xenium spatial transcriptomics experiment into a single, self-contained `.xenium`-compatible output folder.

## Motivation

A single Xenium run often captures many tissue sections (pucks) on one slide. Downstream analysis typically focuses on a subset of those pucks, but the Xenium output files encode *all* pucks in shared coordinate space. Manually extracting individual pucks is error-prone and breaks compatibility with tools like Xenium Explorer and Seurat's `LoadXenium`.

**mosaic** automates this: given a list of puck selection CSVs (exported from Xenium Explorer), it extracts the selected pucks, rearranges them into a configurable grid layout, and produces a complete output folder that Xenium Explorer, Seurat, Scanpy, and other tools can open directly.

## What the pipeline does

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_metadata.py` | Creates output directory, writes updated `experiment.xenium`, copies gene/protein panels and other metadata |
| 02 | `02_cells.py` | Subsets `cells.csv.gz`, `cell_boundaries.csv.gz`, `nucleus_boundaries.csv.gz`; translates coordinates; adds `puck_id`/`puck_name` columns |
| 03 | `03_cells_zarr.py` | Subsets `cells.zarr.zip` (cell masks, labels, summary stats); crops and pastes mask tiles per puck |
| 04 | `04_cells_parquet.py` | Subsets `cells.parquet`, `cell_boundaries.parquet`, `nucleus_boundaries.parquet` |
| 05 | `05_morphology_main.py` | Subsets `morphology.ome.tif` (z-stack); writes pyramidal OME-TIFF with preserved metadata |
| 06 | `06_morphology_focus.py` | Subsets all `morphology_focus/` channels in parallel; preserves multi-channel OME-XML |
| 07 | `07_transcripts.py` | Subsets `transcripts.csv.gz`, `transcripts.parquet`, and `transcripts.zarr.zip` (tiled multi-level format) |
| 08 | `08_analysis.py` | Subsets `analysis.zarr.zip` (cluster assignments for graphclust and kmeans); preserves original cluster labels |

Supporting files:
- **`puck_helpers.py`** — Shared library: puck CSV parsing, grid layout engine, coordinate translation, spatial filtering utilities
- **`config.py`** — Central configuration (overridden at runtime by your project's `run_pipeline.py`)

## Prerequisites

### Software

- Python 3.9+
- Required packages:
  ```
  pip install numpy pandas tifffile imagecodecs pyarrow zarr
  ```

### Input data

1. **Xenium output directory** — The original output folder from a Xenium run (e.g., `output-XETG00201__...`). This is read-only; the pipeline never modifies it.

2. **Puck selection CSVs** — One CSV per puck, exported from Xenium Explorer's cell selection tool. Each CSV should contain a `Cell ID` column listing the cell IDs belonging to that puck. The file may optionally include a comment line `# Selection name : <name>` which the pipeline uses as the puck label.

   Example format:
   ```
   # Selection name : My_Puck_A
   Cell ID
   aaabbbcc-1
   ddeeffgg-2
   ...
   ```

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/mosaic.git
```

### 2. Create a project directory

Create a working directory for your specific experiment. Place your puck selection CSVs here.

```
my_project/
  run_pipeline.py
  puck_A.csv
  puck_B.csv
  puck_C.csv
  ...
```

### 3. Write your `run_pipeline.py`

This is the only file you need to write. It configures paths and layout, then runs the pipeline. Use the template below:

```python
"""
run_pipeline.py — Project-specific pipeline runner.
"""

import sys
from pathlib import Path

# Point to the mosaic pipeline scripts
PIPELINE_DIR = Path("/path/to/mosaic")
sys.path.insert(0, str(PIPELINE_DIR))

import config

# ── Configure ──────────────────────────────────────────────────────────────
CSV_DIR = Path(__file__).parent

config.ORIG = Path("/path/to/xenium/output-XETG00201__...")

config.PUCK_CSVS = [
    CSV_DIR / "puck_A.csv",   # index 0
    CSV_DIR / "puck_B.csv",   # index 1
    CSV_DIR / "puck_C.csv",   # index 2
    CSV_DIR / "puck_D.csv",   # index 3
    CSV_DIR / "puck_E.csv",   # index 4
]

# Grid layout: list of lists, each sub-list is a row of puck indices.
# Indices refer to position in PUCK_CSVS (0-based).
# None or "row" = single row with all pucks side by side.
config.LAYOUT = [[0, 1], [2, 3, 4]]    # 2 rows: top has 2 pucks, bottom has 3

config.OUT = CSV_DIR / "Xenium_subset"

config.PUCK_GAP_UM = 500   # gap between pucks in microns

config.MORPH_WORKERS = 0   # 0 = auto (one worker per CPU core / focus channel)

# ── Run ────────────────────────────────────────────────────────────────────
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
        "08_analysis",
    ]

    for step_name in steps:
        print(f"\n{'='*70}")
        print(f"  RUNNING: {step_name}")
        print(f"{'='*70}\n")
        mod = importlib.import_module(step_name)
        mod.main()
        print()
```

### 4. Run

```bash
cd my_project
python run_pipeline.py
```

The output folder (`Xenium_subset/` by default) will be a valid `.xenium`-compatible directory that can be opened directly in Xenium Explorer, loaded with `Seurat::LoadXenium()`, or read by Scanpy/Squidpy.

## Configuration reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `ORIG` | `Path` | Path to the original Xenium output directory (read-only) |
| `PUCK_CSVS` | `list[Path]` | Ordered list of puck selection CSV paths |
| `LAYOUT` | `list[list[int]]` or `None` | Grid arrangement of pucks. `None` = single row. Each sub-list is a row of 0-based puck indices |
| `OUT` | `Path` | Output directory for the subset |
| `PUCK_GAP_UM` | `float` | Spacing between pucks in microns (default: 500) |
| `MORPH_WORKERS` | `int` | Number of parallel workers for morphology_focus processing. 0 = auto |

## Layout examples

```python
# All pucks in a single row (default)
config.LAYOUT = None

# Vertical stack
config.LAYOUT = [[0], [1], [2]]

# 2x2 grid
config.LAYOUT = [[0, 1], [2, 3]]

# 3 rows with different widths (2-5-5 layout)
config.LAYOUT = [[0, 1], [2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]
```

Every puck index (0 through N-1) must appear exactly once across all rows.

## Output contents

The output directory mirrors the structure of a standard Xenium output folder:

```
Xenium_subset/
  experiment.xenium          # updated num_cells
  gene_panel.json
  protein_panel.json
  cells.csv.gz               # subset + translated, with puck_id/puck_name columns
  cells.parquet
  cells.zarr.zip             # cell masks, labels, summary
  cell_boundaries.csv.gz
  cell_boundaries.parquet
  nucleus_boundaries.csv.gz
  nucleus_boundaries.parquet
  transcripts.csv.gz
  transcripts.parquet
  transcripts.zarr.zip       # tiled multi-level transcript data
  morphology.ome.tif         # pyramidal OME-TIFF z-stack
  morphology_focus/          # per-channel focus images
  analysis.zarr.zip          # cluster assignments (graphclust, kmeans)
  analysis/                  # stub HTML files
  analysis_summary.html
  puck_manifest.csv          # puck_id -> puck_name mapping
```

## Notes

- The pipeline adds a 150 um margin around each puck's bounding box to capture boundary cells and surrounding morphology context.
- Morphology images and cell masks are precisely aligned through a shared coordinate translation system.
- Overlapping puck boundaries are resolved by nearest-center assignment to prevent duplicate cells.
- Cluster assignments from the original Xenium analysis are preserved with their original labels; clusters that lose all members in the subset remain in the schema as empty entries.
- The output `cells.csv.gz` and `cells.parquet` include `puck_id` and `puck_name` columns for easy per-puck filtering in downstream analysis.
