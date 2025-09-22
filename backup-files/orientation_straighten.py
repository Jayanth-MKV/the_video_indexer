"""Efficient, correct EXIF orientation straightening for images (JPEG/TIFF/HEIC/etc.).

Why this file?
The previous demo used a cascading mutation pattern to decide transforms. This version:
  - Reads EXIF orientation (value 1..8) via exifread (fallback to Pillow if available)
  - Applies the minimal exact transform sequence
  - Skips work if orientation is already normal (1)
  - Optionally uses Pillow's ImageOps.exif_transpose (fast path) if it yields a change

EXIF Orientation reference (1..8):
 1 = Normal
 2 = Mirrored horizontal         (flip left-right)
 3 = Rotated 180°
 4 = Mirrored vertical           (flip top-bottom)
 5 = Mirrored horizontal then rotated 270° CW (i.e. 90° CCW)
 6 = Rotated 90° CW              (Pillow ROTATE_270)
 7 = Mirrored horizontal then rotated 90° CW  (i.e. 270° CCW)
 8 = Rotated 270° CW             (Pillow ROTATE_90)

Usage examples (PowerShell):
  uv run orientation_straighten.py Pic.HEIC
  uv run orientation_straighten.py photo.jpg --inplace
  uv run orientation_straighten.py *.jpg -o corrected

Outputs:
  - By default writes <stem>_oriented<ext> next to the original unless --inplace or --output-dir specified.
  - Keeps original format unless saving HEIC (then converts to PNG unless plugin supports writing).
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
from typing import Iterable, Optional

import exifread
from PIL import Image, ImageOps

try:  # Enable HEIC/HEIF/AVIF if pillow-heif installed
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

# Map orientation integer to a sequence of Pillow operations (callables that take & return Image)
# We'll implement composite orientations (5 and 7) as two steps.
from functools import partial


def _flip_lr(im: Image.Image) -> Image.Image:
    return im.transpose(Image.FLIP_LEFT_RIGHT)


def _flip_tb(im: Image.Image) -> Image.Image:
    return im.transpose(Image.FLIP_TOP_BOTTOM)


def _rot_90(im: Image.Image) -> Image.Image:  # 90° CCW
    return im.transpose(Image.ROTATE_90)


def _rot_180(im: Image.Image) -> Image.Image:
    return im.transpose(Image.ROTATE_180)


def _rot_270(im: Image.Image) -> Image.Image:  # 90° CW
    return im.transpose(Image.ROTATE_270)

ORIENTATION_TRANSFORMS = {
    1: (),  # no-op
    2: (_flip_lr,),
    3: (_rot_180,),
    4: (_flip_tb,),
    5: (_flip_lr, _rot_90),   # mirror horizontal + rotate 90 CCW
    6: (_rot_270,),           # rotate 90 CW
    7: (_flip_lr, _rot_270),  # mirror horizontal + rotate 90 CW
    8: (_rot_90,),            # rotate 270 CW (i.e. 90 CCW) -> wait: spec says 270 CW => 90 CCW
}


def read_orientation_exifread(path: Path) -> Optional[int]:
    """Return orientation (1..8) using exifread, or None if not found."""
    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="Image Orientation")
    except Exception as e:  # pragma: no cover
        logging.debug("exifread failed for %s: %s", path, e)
        return None
    tag = tags.get("Image Orientation") if tags else None
    if not tag:
        return None
    # exifread IfdTag -> try .values (list) or str
    values = getattr(tag, "values", None)
    if values and isinstance(values, (list, tuple)) and values:
        try:
            v = int(values[0])
            if 1 <= v <= 8:
                return v
        except Exception:
            return None
    try:
        v = int(str(tag))
        if 1 <= v <= 8:
            return v
    except Exception:
        pass
    return None


def read_orientation_pillow(im: Image.Image) -> Optional[int]:
    """Attempt to read orientation via Pillow's internal EXIF (if JPEG/TIFF)."""
    try:
        exif = im.getexif()
        if not exif:
            return None
        # EXIF orientation tag code is 0x0112 == 274
        v = exif.get(274)
        if isinstance(v, int) and 1 <= v <= 8:
            return v
    except Exception:  # pragma: no cover
        return None
    return None


def determine_orientation(path: Path, im: Image.Image) -> int:
    orient = read_orientation_exifread(path)
    if orient:
        return orient
    orient = read_orientation_pillow(im)
    return orient or 1


def apply_orientation(im: Image.Image, orientation: int) -> Image.Image:
    funcs = ORIENTATION_TRANSFORMS.get(orientation, ())
    for f in funcs:
        im = f(im)
    return im


def straighten_image(path: Path) -> tuple[Image.Image, int, int]:
    """Load image and return (possibly transformed_image, original_orientation, applied_orientation).

    applied_orientation is the orientation we acted on (1..8). If 1, the returned image is original.
    """
    im = Image.open(path)

    # Fast path: try Pillow's ImageOps.exif_transpose (works for common formats) but capture original.
    original = im
    orientation = determine_orientation(path, im)
    if orientation == 1:
        return im, orientation, orientation

    # We'll rely on our own mapping to keep behavior explicit.
    im2 = apply_orientation(im, orientation)
    return im2, orientation, orientation


def save_image(im: Image.Image, src: Path, output_dir: Optional[Path], inplace: bool) -> Path:
    if inplace:
        # Overwrite: preserve format (except some like HEIC may not save; fallback PNG)
        out_path = src
    else:
        dest_dir = output_dir or src.parent
        out_path = dest_dir / f"{src.stem}_oriented{src.suffix}"
    # For safety: if format unsupported for saving (e.g. HEIC), convert to PNG
    fmt = (im.format or src.suffix.lstrip('.')).upper() if hasattr(im, 'format') else src.suffix.lstrip('.')
    try:
        im.save(out_path)
    except Exception:
        # fallback to PNG
        out_path = out_path.with_suffix('.png')
        im.save(out_path, format='PNG')
    return out_path


def iter_paths(patterns: Iterable[str]) -> Iterable[Path]:
    for pat in patterns:
        p = Path(pat)
        if any(ch in pat for ch in "*?[]"):
            # glob pattern
            for match in p.parent.glob(p.name):
                if match.is_file():
                    yield match
        else:
            if p.is_file():
                yield p


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Straighten images using EXIF orientation.")
    parser.add_argument("paths", nargs="+", help="Image file(s) or glob patterns")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original files in place")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory to write corrected images")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args(argv[1:])

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING, format="%(message)s")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    seen_any = False
    for path in iter_paths(args.paths):
        seen_any = True
        try:
            img, orig_orient, applied = straighten_image(path)
            if orig_orient == 1:
                logging.info("%s: orientation=1 (normal) -> no change", path.name)
                if not args.inplace:
                    # Optionally still copy if output dir specified
                    if args.output_dir:
                        out_path = save_image(img, path, args.output_dir, inplace=False)
                        logging.info("  copied -> %s", out_path.name)
                continue
            out_path = save_image(img, path, args.output_dir, args.inplace)
            logging.info("%s: orientation=%d -> corrected -> %s", path.name, orig_orient, out_path.name)
        except Exception as e:
            logging.error("%s: error: %s", path, e)
            exit_code = 1
    if not seen_any:
        logging.error("No matching files")
        return 2
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
