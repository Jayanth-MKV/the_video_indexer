"""Small standalone demo for using the `exifread` library.

Usage (Windows PowerShell examples):
  uv run exifread_demo.py path/to/image.jpg
  uv run exifread_demo.py image1.png Pic.HEIC  # (likely yields few or no tags for PNG/HEIC)

Notes:
- exifread focuses on EXIF data typically present in JPEG/TIFF (sometimes in some RAW formats).
- It does NOT decode general textual metadata chunks in PNG, nor HEIC container structure. For HEIC
  you usually need other tooling (e.g., pillow-heif + Pillow as shown in `test.py`).
- We skip very large binary blobs (thumbnails etc.) for readability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import exifread

# Some EXIF tag names that are often most interesting; adjust as needed.
COMMON_TAG_PREFIXES = [
    "Image Make",
    "Image Model",
    "EXIF DateTimeOriginal",
    "EXIF Lens",
    "EXIF FNumber",
    "EXIF ExposureTime",
    "EXIF ISOSpeedRatings",
    "EXIF FocalLength",
    "GPS GPSLatitude",
    "GPS GPSLongitude",
]

# Tags we consider "large" and will truncate / skip printing full value.
LARGE_VALUE_KEYS = {"JPEGThumbnail", "TIFFThumbnail", "MakerNote", "EXIF MakerNote"}
MAX_VALUE_LEN = 200

def read_exif_tags(path: Path):
    """Read EXIF tags from a file using exifread.

    Returns a dict-like of tags. exifread returns an OrderedDict of tag name -> IfdTag.
    """
    with path.open("rb") as fh:
        # details: stop_tag can limit reading early; details, strict control errors.
        tags = exifread.process_file(fh, details=False)
    return tags


def print_tags(path: Path) -> None:
    try:
        tags = read_exif_tags(path)
    except Exception as e:  # broad for demo purposes
        print(f"{path.name}: error reading EXIF via exifread: {e}")
        return

    if not tags:
        print(f"{path.name}: no EXIF tags found (or unsupported format).")
        return

    print(f"{path.name}: {len(tags)} EXIF tag(s)")

    # First show a curated subset if present.
    shown_subset = False
    for prefix in COMMON_TAG_PREFIXES:
        if prefix in tags:
            val = tags[prefix]
            print(f"  {prefix}: {val}")
            shown_subset = True
    if shown_subset:
        print("  -- (subset above, full list below) --")

    # Then show all (excluding already shown) with truncation for huge values
    for key, value in tags.items():
        if key in COMMON_TAG_PREFIXES:
            continue
        if key in LARGE_VALUE_KEYS:
            print(f"  {key}: <{len(str(value))} bytes omitted>")
            continue
        text = str(value)
        if len(text) > MAX_VALUE_LEN:
            text = text[: MAX_VALUE_LEN - 3] + "..."
        print(f"  {key}: {text}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Provide at least one image file path. Example: uv run exifread_demo.py photo.jpg")
        return 2

    exit_code = 0
    for name in argv[1:]:
        p = Path(name)
        if not p.exists():
            print(f"Skipping {name}: file not found")
            exit_code = 1
            continue
        print_tags(p)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
