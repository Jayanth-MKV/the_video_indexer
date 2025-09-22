"""Auto-straighten images by detecting building lines, horizons, and vertical structures.

This goes beyond EXIF orientation to actually analyze image content and correct:
  1. Camera tilt (horizon not level)
  2. Perspective distortion from buildings/structures
  3. Rotation needed to make vertical lines truly vertical

Uses OpenCV for:
  - Edge detection (Canny)
  - Line detection (Hough lines)
  - Perspective correction
  - Rotation correction

Installation:
  uv add opencv-python

Usage:
  uv run auto_straighten.py Pic.HEIC --save-debug
  uv run auto_straighten.py image.jpg --method=lines
  uv run auto_straighten.py *.jpg --output-dir=straightened

Methods:
  - horizon: detect horizontal lines (good for landscapes)
  - lines: detect dominant vertical/horizontal lines (buildings)  
  - combined: try both approaches and pick best result
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    # PIL is RGB, OpenCV expects BGR
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) back to PIL Image."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def detect_horizon_lines(img: np.ndarray, debug: bool = False) -> Optional[float]:
    """Detect horizon/horizontal lines and return rotation angle in degrees."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with more sensitive parameters
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    # Detect lines with lower threshold for subtle lines
    lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=50)  # Higher angle resolution, lower threshold
    
    if lines is None:
        logging.warning("No lines detected for horizon method")
        return None
    
    # Filter for horizontal-ish lines (within 45 degrees of horizontal)
    horizontal_angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = math.degrees(theta) - 90  # Convert to degrees relative to horizontal
        if abs(angle_deg) < 45:  # Wider range for more detection
            horizontal_angles.append(angle_deg)
    
    if not horizontal_angles:
        logging.warning("No horizontal lines found")
        return None
    
    # Return median angle to correct (with precision to 0.1 degree)
    median_angle = np.median(horizontal_angles)
    if abs(median_angle) < 0.1:
        median_angle = 0.0  # Round very small angles to zero
    
    logging.info(f"Detected horizon tilt: {median_angle:.1f} degrees from {len(horizontal_angles)} lines")
    
    if debug:
        # Draw detected lines for visualization with proper coordinate handling
        line_img = img.copy()
        height, width = img.shape[:2]
        line_count = 0
        
        for rho, theta in lines[:, 0]:
            angle_deg = math.degrees(theta) - 90
            if abs(angle_deg) < 45:  # Show wider range of lines
                # Calculate line endpoints properly
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Calculate intersection points with image boundaries
                if abs(a) > 0.001:  # Non-vertical line
                    # Find intersections with left and right edges
                    x1, x2 = 0, width - 1
                    y1 = int(y0 + (x1 - x0) * (b/a))
                    y2 = int(y0 + (x2 - x0) * (b/a))
                else:  # Nearly vertical line
                    # Find intersections with top and bottom edges
                    y1, y2 = 0, height - 1
                    x1 = int(x0 + (y1 - y0) * (a/b))
                    x2 = int(x0 + (y2 - y0) * (a/b))
                
                # Clip to image boundaries and draw
                x1 = max(0, min(width - 1, x1))
                x2 = max(0, min(width - 1, x2))
                y1 = max(0, min(height - 1, y1))
                y2 = max(0, min(height - 1, y2))
                
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
                line_count += 1
                
                # Limit lines to avoid clutter
                if line_count >= 50:
                    break
        
        cv2.imwrite("debug_horizon_lines.jpg", line_img)
        logging.info(f"Saved debug_horizon_lines.jpg with {line_count} lines drawn")
    
    return -median_angle  # Negative to correct the tilt


def detect_vertical_lines(img: np.ndarray, debug: bool = False) -> Optional[float]:
    """Detect vertical building lines and return rotation angle needed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced edge detection for buildings
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines with more sensitive parameters
    lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=40)  # Even more sensitive
    
    if lines is None:
        logging.warning("No lines detected for vertical method")
        return None
    
    # Filter for vertical-ish lines (wider range)
    vertical_angles = []
    def calculate_rotation_from_vertical(theta: float) -> float:
        """Calculate rotation needed to make a line vertical given its theta."""
        rotation_needed = math.degrees(theta) - 90
        if rotation_needed > 45:
            rotation_needed -= 90
        elif rotation_needed < -45:
            rotation_needed += 90
        return rotation_needed

    for rho, theta in lines[:, 0]:
        # Convert theta to angle from vertical (0 = perfectly vertical)
        angle_from_vertical = abs(math.degrees(theta) - 90)
        if angle_from_vertical > 90:
            angle_from_vertical = 180 - angle_from_vertical
            
        if angle_from_vertical < 45:  # Within 45 degrees of vertical (wider range)
            # Calculate rotation needed to make this line vertical
            rotation_needed = calculate_rotation_from_vertical(theta)
            vertical_angles.append(rotation_needed)
    
    if not vertical_angles:
        logging.warning("No vertical lines found")
        return None
    
    # Use median to avoid outliers (with precision)
    median_rotation = np.median(vertical_angles)
    if abs(median_rotation) < 0.1:
        median_rotation = 0.0
    
    logging.info(f"Detected building tilt: {median_rotation:.1f} degrees from {len(vertical_angles)} lines")
    
    if debug:
        # Draw detected vertical lines with proper bounds checking
        line_img = img.copy()
        height, width = img.shape[:2]
        line_count = 0
        
        for rho, theta in lines[:, 0]:
            angle_from_vertical = abs(math.degrees(theta) - 90)
            if angle_from_vertical > 90:
                angle_from_vertical = 180 - angle_from_vertical
            if angle_from_vertical < 45:  # Show wider range
                # Calculate line endpoints more carefully
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Calculate line endpoints that span the image properly
                if abs(b) > 0.001:  # Non-horizontal line
                    # Calculate x coordinates at top and bottom of image
                    y1, y2 = 0, height - 1
                    x1 = int(x0 + (y1 - y0) * (-a/b))
                    x2 = int(x0 + (y2 - y0) * (-a/b))
                else:  # Nearly horizontal line
                    # Calculate y coordinates at left and right of image
                    x1, x2 = 0, width - 1
                    y1 = int(y0 + (x1 - x0) * (-b/a))
                    y2 = int(y0 + (x2 - x0) * (-b/a))
                
                # Only draw if endpoints are reasonable
                if (-width <= x1 <= 2*width and -height <= y1 <= 2*height and 
                    -width <= x2 <= 2*width and -height <= y2 <= 2*height):
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines
                    line_count += 1
                    
                    # Limit number of lines drawn to avoid clutter
                    if line_count >= 50:
                        break
        
        cv2.imwrite("debug_vertical_lines.jpg", line_img)
        logging.info(f"Saved debug_vertical_lines.jpg with {line_count} lines drawn")
    
    return -median_rotation  # Negative to correct


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle in degrees."""
    if abs(angle) < 0.1:  # Skip tiny rotations
        return img
        
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos_angle = abs(matrix[0, 0])
    sin_angle = abs(matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust the rotation matrix to account for translation
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform rotation with white background
    rotated = cv2.warpAffine(img, matrix, (new_width, new_height), 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated


def auto_straighten_image(pil_img: Image.Image, method: str = "combined", debug: bool = False) -> Image.Image:
    """Auto-straighten image using specified method."""
    cv_img = pil_to_cv2(pil_img)
    
    angles = []
    
    if method in ["horizon", "combined"]:
        horizon_angle = detect_horizon_lines(cv_img, debug)
        if horizon_angle is not None:
            angles.append(("horizon", horizon_angle))
    
    if method in ["lines", "combined"]:
        vertical_angle = detect_vertical_lines(cv_img, debug)
        if vertical_angle is not None:
            angles.append(("vertical", vertical_angle))
    
    if not angles:
        logging.warning("No correction angles detected")
        return pil_img
    
    # Choose best angle
    if len(angles) == 1:
        chosen_method, chosen_angle = angles[0]
    else:
        # For combined method, prefer the smaller correction (more conservative)
        angles.sort(key=lambda x: abs(x[1]))
        chosen_method, chosen_angle = angles[0]
        logging.info(f"Combined method: chose {chosen_method} angle {chosen_angle:.2f}° over others")
    
    logging.info(f"Applying {chosen_method} correction: {chosen_angle:.2f} degrees")
    
    # Apply rotation
    straightened = rotate_image(cv_img, chosen_angle)
    
    return cv2_to_pil(straightened)


def process_image(input_path: Path, output_dir: Optional[Path], method: str, debug: bool, manual_angle: Optional[float]) -> None:
    """Process a single image file."""
    try:
        # Load image
        pil_img = Image.open(input_path)
        logging.info(f"Processing {input_path.name} ({pil_img.size})")
        
        if manual_angle is not None:
            # Manual override
            cv_img = pil_to_cv2(pil_img)
            straightened = rotate_image(cv_img, manual_angle)
            result = cv2_to_pil(straightened)
            logging.info(f"Applied manual rotation: {manual_angle} degrees")
        else:
            # Auto-straighten
            result = auto_straighten_image(pil_img, method, debug)
        
        # Determine output path
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_straightened.jpg"
        else:
            output_path = input_path.with_name(f"{input_path.stem}_straightened.jpg")
        
        # Save result
        result.save(output_path, "JPEG", quality=95)
        logging.info(f"Saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Auto-straighten images by detecting building/horizon lines")
    parser.add_argument("images", nargs="+", help="Image files to process")
    parser.add_argument("--method", choices=["horizon", "lines", "combined"], default="combined",
                       help="Detection method (default: combined)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--save-debug", action="store_true", help="Save debug images showing detected lines")
    parser.add_argument("--angle", type=float, help="Manual rotation angle in degrees (overrides auto-detection)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    
    args = parser.parse_args(argv[1:])
    
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            logging.error(f"File not found: {path}")
            continue
        
        process_image(path, args.output_dir, args.method, args.save_debug, args.angle)
    
    return 0


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV not installed. Run: uv add opencv-python")
        sys.exit(1)
    
    raise SystemExit(main(sys.argv))
