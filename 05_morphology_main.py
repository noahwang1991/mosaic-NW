"""
05_morphology_main.py — Subset morphology.ome.tif (the z-stack).

Reads the multi-plane z-stack, crops each puck region, pastes onto
a new canvas, writes as pyramidal OME-TIFF matching the original format.
Preserves the full OME-XML metadata (channel names, instrument, annotations).
"""

import gc
import re
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    import tifffile
except ImportError:
    sys.exit("pip install tifffile imagecodecs")

CROP_MARGIN_UM = 150.0


class _Crop:
    pass


def _compute_crops(bboxes, puck_stats, orig_H, orig_W, pixel_size):
    crops = []
    for bb, row in zip(bboxes, puck_stats.itertuples()):
        pc = _Crop()
        pc.name   = bb["puck_name"]
        x_lo = row.x_min - CROP_MARGIN_UM; x_hi = row.x_max + CROP_MARGIN_UM
        y_lo = row.y_min - CROP_MARGIN_UM; y_hi = row.y_max + CROP_MARGIN_UM
        pc.src_c0 = max(0, int(np.floor(x_lo / pixel_size)))
        pc.src_c1 = min(orig_W, int(np.ceil(x_hi / pixel_size)))
        pc.src_r0 = max(0, int(np.floor(y_lo / pixel_size)))
        pc.src_r1 = min(orig_H, int(np.ceil(y_hi / pixel_size)))
        paste_c   = round(bb["x_min_new"] / pixel_size)
        paste_r   = round(bb["y_min_new"] / pixel_size)
        pc.skip_c = max(0, -paste_c)
        pc.skip_r = max(0, -paste_r)
        pc.dst_c  = max(0, paste_c)
        pc.dst_r  = max(0, paste_r)
        crops.append(pc)
    return crops


def _pyramid_levels(w, h):
    levels = []
    while min(w, h) > 512:
        w //= 2; h //= 2
        levels.append((w, h))
    return levels


def _update_ome_xml(orig_xml, new_w, new_h):
    xml = orig_xml
    xml = re.sub(r'SizeX="\d+"', f'SizeX="{new_w}"', xml)
    xml = re.sub(r'SizeY="\d+"', f'SizeY="{new_h}"', xml)
    new_uuid = f"urn:uuid:{uuid.uuid4()}"
    xml = re.sub(r'UUID="urn:uuid:[^"]*"', f'UUID="{new_uuid}"', xml, count=1)
    xml = xml.replace('µm', 'um')
    xml = xml.encode('ascii', 'xmlcharrefreplace').decode('ascii')
    return xml


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM
    from puck_helpers import build_puck_map, PIXEL_SIZE

    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )

    max_new_x = max(bb["x_max_new"] for bb in bboxes)
    max_new_y = max(bb["y_max_new"] for bb in bboxes)
    CANVAS_W  = int(np.ceil(max_new_x / PIXEL_SIZE))
    CANVAS_H  = int(np.ceil(max_new_y / PIXEL_SIZE))
    print(f"Canvas: {CANVAS_H} rows × {CANVAS_W} cols")

    src_path = ORIG / "morphology.ome.tif"
    dst_path = OUT / "morphology.ome.tif"

    with tifffile.TiffFile(str(src_path)) as tif:
        n_pages = len(tif.pages)
        orig_H  = tif.pages[0].shape[0]
        orig_W  = tif.pages[0].shape[1]
        dtype   = tif.pages[0].dtype
        orig_ome_xml = tif.ome_metadata
        print(f"\nSource: {n_pages} pages  ({orig_H}×{orig_W})  dtype={dtype}")

    crops = _compute_crops(bboxes, puck_stats, orig_H, orig_W, PIXEL_SIZE)
    for pc in crops:
        print(f"  [{pc.name}] src r:[{pc.src_r0},{pc.src_r1}] c:[{pc.src_c0},{pc.src_c1}]  "
              f"dst r={pc.dst_r} c={pc.dst_c}")

    out_shape = (n_pages, CANVAS_H, CANVAS_W) if n_pages > 1 else (CANVAS_H, CANVAS_W)
    all_planes = np.zeros(out_shape, dtype=dtype)
    print(f"\nOutput shape: {all_planes.shape}  ({all_planes.nbytes / 1e9:.2f} GB)")

    with tifffile.TiffFile(str(src_path)) as tif:
        for p in range(n_pages):
            print(f"  Plane {p+1}/{n_pages} …", flush=True)
            plane  = tif.pages[p].asarray()
            canvas = all_planes if n_pages == 1 else all_planes[p]
            canvas[:] = 0
            for pc in crops:
                raw = plane[pc.src_r0:pc.src_r1, pc.src_c0:pc.src_c1]
                crop = raw[pc.skip_r:, pc.skip_c:]
                hh = min(crop.shape[0], CANVAS_H - pc.dst_r)
                ww = min(crop.shape[1], CANVAS_W - pc.dst_c)
                if hh > 0 and ww > 0:
                    canvas[pc.dst_r:pc.dst_r+hh, pc.dst_c:pc.dst_c+ww] = crop[:hh, :ww]
            del plane; gc.collect()
            print(f"    max={int(canvas.max())}  nnz={np.count_nonzero(canvas):,}")

    sub_levels = _pyramid_levels(CANVAS_W, CANVAS_H)
    print(f"\nPyramid: 1 full + {len(sub_levels)} sub-levels")

    y_ax = len(all_planes.shape) - 2
    x_ax = len(all_planes.shape) - 1
    sub_opts = dict(tile=(512, 512), compression="deflate",
                    photometric="minisblack", metadata=None)

    ome_xml = _update_ome_xml(orig_ome_xml, CANVAS_W, CANVAS_H) if orig_ome_xml else None

    if dst_path.exists():
        dst_path.unlink()

    print(f"\nWriting {dst_path.name} …", flush=True)
    with tifffile.TiffWriter(str(dst_path), bigtiff=True) as tw:
        tw.write(all_planes, subifds=len(sub_levels),
                 tile=(512, 512), compression="deflate", photometric="minisblack",
                 description=ome_xml, metadata=None)
        level_data = all_planes
        for lw, lh in sub_levels:
            sl = [slice(None)] * len(level_data.shape)
            sl[y_ax] = slice(None, None, 2)
            sl[x_ax] = slice(None, None, 2)
            level_data = level_data[tuple(sl)]
            tw.write(level_data, subfiletype=1, **sub_opts)

    del all_planes; gc.collect()
    sz = dst_path.stat().st_size / 1e6
    print(f"\n✓ Written: {dst_path.name}  ({sz:.0f} MB)")
    print("✓ 05_morphology_main.py complete.")


if __name__ == "__main__":
    main()
