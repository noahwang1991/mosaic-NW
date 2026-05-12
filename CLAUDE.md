# mosaic — Xenium Puck-Subset Pipeline

## Project overview

This is a reusable pipeline that subsets and rearranges tissue pucks from a 10x Xenium spatial transcriptomics experiment. Given puck selection CSVs (exported from Xenium Explorer), it produces a complete `.xenium`-compatible output folder openable by Xenium Explorer, Seurat `LoadXenium()`, and Scanpy.

**Repository**: `mosaic` on GitHub.

## User

Spatial transcriptomics researcher. Python-fluent. Prefers running scripts themselves — do NOT execute long-running pipeline steps. Validate by reading code, suggest commands for user to run. Working on an external T5 EVO drive.

## File structure

```
PyScripts/                  # Pipeline scripts (project-agnostic)
  config.py                 # Central config; overridden by each project's run_pipeline.py
  puck_helpers.py           # Shared: puck CSV parsing, grid layout, coordinate translation
  01_metadata.py            # experiment.xenium, gene/protein panels, verbatim metadata
  02_cells.py               # cells.csv.gz, cell/nucleus_boundaries.csv.gz + puck_id columns
  03_cells_zarr.py          # cells.zarr.zip (masks, labels, summary), cell_feature_matrix
  04_cells_parquet.py       # cells.parquet, boundaries.parquet
  05_morphology_main.py     # morphology.ome.tif (z-stack, pyramidal OME-TIFF)
  06_morphology_focus.py    # morphology_focus/ channels (parallel, auto-discovered)
  07_transcripts.py         # transcripts.csv.gz, .parquet, .zarr.zip (tiled multi-level)
  08_analysis.py            # analysis.zarr.zip (cluster assignments: graphclust + kmeans)
```

Project-specific runner (NOT in PyScripts, lives in each project folder):
```
bioengineering_xenium_puck_csvs/
  run_pipeline.py           # Overrides config with project paths, puck CSVs, layout
  csv0.csv ... csv11.csv    # Puck selection CSVs
  Xenium_subset/            # Output folder
```

Original Xenium data (read-only):
```
/Volumes/T5 EVO/Xenium_Raredon/RQ41335-001_Raredon-849-1/
  output-XETG00201__0073923__Region_1__20260402__174410/
```

## Architecture decisions

### Pipeline design
- All 01–08 scripts and helpers are project-agnostic. Zero hardcoded paths.
- Each project writes its own `run_pipeline.py` that imports `config` and overrides `ORIG`, `PUCK_CSVS`, `LAYOUT`, `OUT`, etc.
- `run_pipeline.py` calls each step's `main()` sequentially via `importlib`.
- `02_cells.py` writes `.keep_idx.npy` and `.cell_ids.txt` to OUT — used by steps 03, 04, 08.

### Grid layout (puck_helpers.py)
- `LAYOUT = [[0,1], [2,3,4,5,6], [7,8,9,10,11]]` = 3 rows (2-5-5).
- Within-row: pucks advance along Y. Between rows: advance along X in reverse so row 0 gets largest X.
- In Xenium Explorer, rotate 270 degrees CW to see intended layout.
- Floor correction ensures all new coordinates >= 0 after translation.

### Coordinate translation
- Each puck gets `dx`, `dy` (microns) computed from grid layout.
- `get_dx_dy()` uses nearest-puck-center assignment when bboxes overlap.
- `dx_dy_for_cells()` uses cell_id → puck_map for exact assignment.
- 150 um margin (`MARGIN_UM`) expands bboxes for spatial filtering (transcripts, boundaries).

### Critical: morphology/mask paste position
- When crop margin extends past original image boundary, source crop start is clamped to 0.
- Paste position MUST be: `src_start + round(dx / PIXEL_SIZE)`, NOT `round(new_min / PIXEL_SIZE)`.
- The difference can be hundreds of pixels for pucks near image edges.
- This is implemented in 03_cells_zarr.py, 05_morphology_main.py, 06_morphology_focus.py.

### Mask puck membership filtering (03_cells_zarr.py)
- When puck bboxes overlap in original space, mask tiles can contain cells from neighboring pucks.
- `orig_idx_to_puck` array maps original cell index → puck index.
- During mask crop, only cells belonging to the current puck are kept; others are zeroed out.

### Format matching
- zarr: `dimension_separator="."`, `order="F"`, zstd compressor, `.zattrs` on tile arrays.
- Morphology: pyramidal OME-TIFF, bigtiff, 512x512 tiles, deflate compression.
- OME-XML: preserve full multi-channel metadata, update SizeX/SizeY, generate new UUIDs.
- `PIXEL_SIZE = 0.2125` um/pixel for morphology and masks.

### analysis.zarr.zip (08_analysis.py)
- Contains `cell_groups/` with 20 groupings (graphclust + kmeans 2-10, for gene and protein expression).
- CSC-like format: `indices` (uint32, 0-indexed cell positions) + `indptr` (uint32, N_clusters+1).
- Subset remaps cell indices via `orig_to_new` from `keep_idx`; empty clusters stay in schema.
- `01_metadata.py` does NOT copy `analysis.zarr.zip` verbatim anymore (removed from VERBATIM list).
- `08_analysis.py` also accepts CLI args: `python 08_analysis.py <ORIG> <OUT>` for standalone use.

### transcripts.zarr.zip (07_transcripts.py)
- Level 0: 9 arrays (full transcript data), sorted by gene_identity, hi-quality first then lo-quality.
- Levels 1+: 4 arrays (cluster_count, gene_identity, gene_offset, location), spatially clustered.
- `gene_offset` shape (N_CODEWORDS, 4): [lo_start, lo_end, hi_start, hi_end].
- Tile grid: `N_Y_FULL=1952, N_X_FULL=852` at level 0; halved at each subsequent level.

## Bugs fixed (for reference if issues recur)

1. **Morphology-to-cell alignment shift**: Pucks near image edges had 100-600px shifts because paste position was computed from unclamped bbox boundary instead of clamped crop start + translation. Fixed in 03, 05, 06.

2. **Bbox overlap ghost cells**: Overlapping puck bboxes caused wrong dx/dy assignment. Fixed with nearest-center assignment in `get_dx_dy()` and puck membership filter in mask crops.

## Constants (puck_helpers.py)

```python
MARGIN_UM     = 150        # bbox expansion for spatial filtering
PIXEL_SIZE    = 0.2125     # um/pixel for morphology and masks
N_CODEWORDS   = 541        # gene panel size
N_Y_FULL      = 1952       # transcript tile grid rows (level 0)
N_X_FULL      = 852        # transcript tile grid cols (level 0)
TILE_SIZES_UM = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
```

## Pending / known issues

- Seurat `LoadXenium` may warn "File not found" for some optional files — non-fatal, cosmetic only.
- `analysis/` directory contains only stub `index.html` files — copied verbatim by 01_metadata.py.
