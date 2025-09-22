"""Demo: read Pic.HEIC, inspect EXIF orientation via exifread, and apply transforms.

Usage:
  uv run orientation_exif_fix.py Pic.HEIC

It will print debug info and save a corrected version next to the original
as Pic_corrected.png (always PNG output just for simplicity).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import exifread
from PIL import Image

# Ensure HEIC support (pillow-heif) is registered if available
try:  # pragma: no cover - best effort
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pass


def _read_img_and_correct_exif_orientation(path: Path) -> Image.Image:
    """Open image, read EXIF orientation with exifread, apply transformations.

    Mirrors the snippet the user provided. Note: EXIF Orientation standard values:
      1 = Normal
      2 = Mirrored horizontal
      3 = Rotated 180
      4 = Mirrored vertical
      5 = Mirrored horizontal then rotated 270 CW
      6 = Rotated 270 CW (a.k.a. 90 CCW)
      7 = Mirrored horizontal then rotated 90 CW
      8 = Rotated 90 CW

    The original snippet mutates the list of values (val += [...]) to cascade
    derived operations. Kept as-is (though alternative would map each directly).
    """
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    im = Image.open(path)

    tags = {}
    with path.open("rb") as file_handle:
        tags = exifread.process_file(file_handle, details=False)

    if "Image Orientation" in tags:
        orientation = tags["Image Orientation"]
        logging.debug("Orientation tag: %s (raw values=%s)", orientation, getattr(orientation, "values", orientation))
        # orientation.values is typically a list/tuple of one int.
        val = list(getattr(orientation, "values", [])) or [int(str(orientation))] if str(orientation).isdigit() else []

        # Original snippet's cascading augmentation logic
        if 2 in val:
            val += [4, 3]
        if 5 in val:
            val += [4, 6]
        if 7 in val:
            val += [4, 8]
        if 3 in val:
            logging.debug("Applying: rotate 180")
            im = im.transpose(Image.ROTATE_180)
        if 4 in val:
            logging.debug("Applying: mirror horizontal")
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if 6 in val:
            logging.debug("Applying: rotate 270 (i.e., 90 CW)")
            im = im.transpose(Image.ROTATE_270)
        if 8 in val:
            logging.debug("Applying: rotate 90 (i.e., 270 CW)")
            im = im.transpose(Image.ROTATE_90)
    else:
        logging.debug("No EXIF orientation tag present.")

    return im


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Provide an image path, e.g.: uv run orientation_exif_fix.py Pic.HEIC")
        return 2

    for name in argv[1:]:
        p = Path(name)
        if not p.exists():
            print(f"Skipping {name}: not found")
            continue
        try:
            im1 = Image.open(p)
            im1.save("initial_image.png", format="PNG")
            corrected = _read_img_and_correct_exif_orientation(p)
            out_path = p.with_name(p.stem + "_corrected.png")
            corrected.save(out_path, format="PNG")
            print(f"Saved corrected image -> {out_path.name}")
        except Exception as e:  # pragma: no cover
            print(f"Error processing {p}: {e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
