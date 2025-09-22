"""Simple metadata reporter that works for PNG, JPEG/TIFF, and HEIC/HEIF/AVIF.

JPEG/TIFF: uses the 'exif' library (richer attribute names).
PNG + HEIC/HEIF/AVIF + others Pillow can open: uses Pillow (with pillow-heif plugin for HEIC family).
"""

from pathlib import Path
from typing import List

from PIL import Image as PILImage
from PIL.ExifTags import TAGS
from exif import Image as ExifImage  # still useful for JPEG/TIFF

# Register HEIF/HEIC/AVIF support if available
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception as _heif_err:  # pragma: no cover - optional dependency failure
    # We'll proceed; HEIC files will fail with a clear message later.
    pass


def describe_exif_with_exif_lib(path: Path) -> None:
    """Handle JPEG/TIFF via exif library."""
    with path.open("rb") as f:
        img = ExifImage(f)
    if img.has_exif:
        print(f"{path.name}: contains EXIF (version {getattr(img, 'exif_version', 'unknown')}) information.")
        # Example: print a small subset of common attributes if present
        for attr in [
            "make",
            "model",
            "datetime_original",
            "f_number",
            "focal_length",
            "exposure_time",
        ]:
            if hasattr(img, attr):
                print(f"  {attr}: {getattr(img, attr)}")
    else:
        print(f"{path.name}: does not contain any EXIF information.")


def describe_exif_with_pillow(path: Path) -> None:
    """Handle PNG (and also works for JPEG) via Pillow.

    For PNG most files won't have EXIF; we also surface textual metadata.
    """
    with PILImage.open(path) as im:
        print(f"{path.name}: format={im.format} size={im.size} mode={im.mode}")
        exif = im.getexif()
        if exif and len(exif) > 0:
            print("  EXIF entries:")
            shown = 0
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                print(f"    {tag}: {value}")
                shown += 1
                if shown >= 15:  # keep output short
                    print("    ... (truncated) ...")
                    break
        else:
            print("  No EXIF data")

        # Show textual metadata (tEXt/iTXt chunks) if any
        if im.info:
            print(im.info)
            text_keys = {k: v for k, v in im.info.items() if isinstance(v, (str, bytes)) and k.lower() not in {"exif"}}
            if text_keys:
                print("  Textual metadata:")
                for k, v in text_keys.items():
                    if isinstance(v, bytes):
                        try:
                            v_disp = v.decode("utf-8", "replace")
                        except Exception:
                            v_disp = str(v)
                    else:
                        v_disp = v
                    if len(v_disp) > 120:
                        v_disp = v_disp[:117] + "..."
                    print(f"    {k}: {v_disp}")


def process_image(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".tif", ".tiff"}:
        describe_exif_with_exif_lib(path)
    else:  # PNG, WEBP, HEIC, etc. (handled by Pillow if plugin present)
        describe_exif_with_pillow(path)


def main(files: List[str]) -> None:
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"Skipping {f}: not found")
            continue
        try:
            process_image(p)
        except Exception as e:  # ensure one bad file doesn't stop the rest
            print(f"Error processing {p.name}: {e}")


if __name__ == "__main__":
    # Change or extend this list as needed
    main(["image1.png", "Pic.HEIC"])