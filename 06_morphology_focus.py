"""
06_morphology_focus.py — Subset all morphology_focus channels in parallel.

Auto-discovers channels by globbing morphology_focus/ch*.ome.tif.
Each channel is processed independently via ProcessPoolExecutor.

Preserves the full multi-channel OME-XML metadata from the original files
so Xenium Explorer sees named channels, instrument info, and annotations.
Also copies morphology_focus/index.html if present.
"""

import gc
import os
import re
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from puck_helpers import PIXEL_SIZE

try:
    import tifffile
except ImportError:
    sys.exit("pip install tifffile imagecodecs")

CROP_MARGIN_UM = 150.0


def _discover_channels(focus_dir: Path) -> list[Path]:
    channels = sorted(focus_dir.glob("ch*.ome.tif"))
    if not channels:
        print(f"  No ch*.ome.tif files found in {focus_dir}")
    return channels


def _pyramid_levels(w, h):
    levels = []
    while min(w, h) > 512:
        w //= 2; h //= 2
        levels.append((w, h))
    return levels


def _build_ome_xml(orig_xml: str, new_w: int, new_h: int,
                   channels: list[Path]) -> dict[str, str]:
    """Build per-file OME-XML from the original multi-channel OME-XML.

    Returns a dict mapping filename → updated OME-XML string.
    SizeX/SizeY are updated. Each file gets a new UUID, and TiffData
    entries are updated to reference the new filenames/UUIDs.
    """
    xml = orig_xml
    xml = re.sub(r'SizeX="\d+"', f'SizeX="{new_w}"', xml)
    xml = re.sub(r'SizeY="\d+"', f'SizeY="{new_h}"', xml)

    file_uuids = {}
    for ch_path in channels:
        file_uuids[ch_path.name] = f"urn:uuid:{uuid.uuid4()}"

    for fname, new_urn in file_uuids.items():
        xml = re.sub(
            rf'(<UUID\s+FileName="{re.escape(fname)}">)urn:uuid:[^<]*(</UUID>)',
            rf'\g<1>{new_urn}\2',
            xml,
        )

    xml = xml.replace('µm', 'um')
    xml = xml.encode('ascii', 'xmlcharrefreplace').decode('ascii')

    result = {}
    for fname, new_urn in file_uuids.items():
        file_xml = re.sub(
            r'(<OME\s[^>]*UUID=")urn:uuid:[^"]*(")',
            rf'\g<1>{new_urn}\2',
            xml,
        )
        result[fname] = file_xml
    return result


def process_channel(
    src_path: Path,
    dst_path: Path,
    bboxes: list,
    puck_stats_records: list,
    canvas_h: int,
    canvas_w: int,
    ome_xml,
):
    """Process a single focus channel. Runs in a worker process."""
    import tifffile as tf
    import numpy as np

    name = src_path.name

    with tf.TiffFile(str(src_path), is_ome=False) as tif:
        n_pages = len(tif.pages)
        orig_H  = tif.pages[0].shape[0]
        orig_W  = tif.pages[0].shape[1]
        dtype   = tif.pages[0].dtype

    crops = []
    for bb, ps in zip(bboxes, puck_stats_records):
        x_lo = ps["x_min"] - CROP_MARGIN_UM
        x_hi = ps["x_max"] + CROP_MARGIN_UM
        y_lo = ps["y_min"] - CROP_MARGIN_UM
        y_hi = ps["y_max"] + CROP_MARGIN_UM
        src_c0 = max(0, int(np.floor(x_lo / PIXEL_SIZE)))
        src_c1 = min(orig_W, int(np.ceil(x_hi / PIXEL_SIZE)))
        src_r0 = max(0, int(np.floor(y_lo / PIXEL_SIZE)))
        src_r1 = min(orig_H, int(np.ceil(y_hi / PIXEL_SIZE)))
        paste_c = src_c0 + round(bb["dx"] / PIXEL_SIZE)
        paste_r = src_r0 + round(bb["dy"] / PIXEL_SIZE)
        skip_c  = max(0, -paste_c)
        skip_r  = max(0, -paste_r)
        dst_c   = max(0, paste_c)
        dst_r   = max(0, paste_r)
        crops.append((src_r0, src_r1, src_c0, src_c1, skip_r, skip_c, dst_r, dst_c))

    out_shape = (n_pages, canvas_h, canvas_w) if n_pages > 1 else (canvas_h, canvas_w)
    all_planes = np.zeros(out_shape, dtype=dtype)

    with tf.TiffFile(str(src_path), is_ome=False) as tif:
        for p in range(n_pages):
            plane = tif.pages[p].asarray()
            canvas = all_planes if n_pages == 1 else all_planes[p]
            canvas[:] = 0
            for (sr0, sr1, sc0, sc1, skr, skc, dr, dc) in crops:
                raw = plane[sr0:sr1, sc0:sc1]
                crop = raw[skr:, skc:]
                hh = min(crop.shape[0], canvas_h - dr)
                ww = min(crop.shape[1], canvas_w - dc)
                if hh > 0 and ww > 0:
                    canvas[dr:dr+hh, dc:dc+ww] = crop[:hh, :ww]
            del plane

    sub_levels = _pyramid_levels(canvas_w, canvas_h)
    y_ax = len(all_planes.shape) - 2
    x_ax = len(all_planes.shape) - 1

    sub_opts = dict(tile=(512, 512), compression="deflate",
                    photometric="minisblack", metadata=None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    with tf.TiffWriter(str(dst_path), bigtiff=True) as tw:
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

    sz = dst_path.stat().st_size / 1e6
    del all_planes
    return f"  ✓ {name}  ({sz:.0f} MB)"


def main():
    from config import PUCK_CSVS, ORIG, OUT, LAYOUT, PUCK_GAP_UM, MORPH_WORKERS
    from puck_helpers import build_puck_map, PIXEL_SIZE

    puck_map, all_cell_ids, puck_stats, bboxes = build_puck_map(
        PUCK_CSVS, ORIG / "cells.csv.gz",
        puck_gap_um=PUCK_GAP_UM, layout=LAYOUT, verbose=False,
    )

    max_new_x = max(bb["x_max_new"] for bb in bboxes)
    max_new_y = max(bb["y_max_new"] for bb in bboxes)
    canvas_w  = int(np.ceil(max_new_x / PIXEL_SIZE))
    canvas_h  = int(np.ceil(max_new_y / PIXEL_SIZE))
    print(f"Canvas: {canvas_h} rows × {canvas_w} cols")

    focus_dir = ORIG / "morphology_focus"
    channels  = _discover_channels(focus_dir)
    print(f"Found {len(channels)} focus channels")

    if not channels:
        return

    out_dir = OUT / "morphology_focus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read original OME-XML and build per-file versions with updated dims
    print("Reading original OME-XML metadata …")
    with tifffile.TiffFile(str(channels[0])) as tif:
        orig_ome_xml = tif.ome_metadata

    if orig_ome_xml:
        ome_per_file = _build_ome_xml(orig_ome_xml, canvas_w, canvas_h, channels)
        print(f"  Built OME-XML for {len(ome_per_file)} channels")
    else:
        ome_per_file = {}
        print("  WARNING: no OME metadata found in original")

    # Copy index.html if present
    idx_src = focus_dir / "index.html"
    if idx_src.exists():
        shutil.copy2(str(idx_src), str(out_dir / "index.html"))
        print("  Copied index.html")

    puck_stats_records = puck_stats.to_dict("records")

    n_workers = MORPH_WORKERS if MORPH_WORKERS > 0 else min(os.cpu_count() or 4, len(channels))
    print(f"Processing with {n_workers} parallel workers …\n")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for ch_path in channels:
            dst = out_dir / ch_path.name
            ch_ome = ome_per_file.get(ch_path.name)
            fut = pool.submit(
                process_channel, ch_path, dst,
                bboxes, puck_stats_records, canvas_h, canvas_w, ch_ome,
            )
            futures[fut] = ch_path.name

        for fut in as_completed(futures):
            ch_name = futures[fut]
            try:
                msg = fut.result()
                print(msg, flush=True)
            except Exception as exc:
                print(f"  ✗ {ch_name}: {exc}", flush=True)

    print(f"\n✓ 06_morphology_focus.py complete. ({len(channels)} channels)")


if __name__ == "__main__":
    main()
