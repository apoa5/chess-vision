"""Image I/O for uint8 grayscale, BGR, and BGRA arrays.

File arguments are relative to the caller's working directory. Loaded images
are always three-channel BGR; coordinates use x=column, y=row.
"""

from pathlib import Path

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> None:
    """Reject empty or unsupported arrays before calling OpenCV."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a nonempty NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError("Image must have uint8 pixel values.")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError("Image must have shape (height, width), (height, width, 3), or (height, width, 4).")


def load_image(path: str | Path) -> np.ndarray:
    """Load a photo as BGR; raise a useful error for missing/undecodable files."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"OpenCV could not decode image: {path}") from error
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def ensure_output_directory(path: str | Path) -> Path:
    """Create an output directory (and its parents) if necessary."""
    directory = Path(path).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(f"Could not create output directory: {directory}") from error
    return directory


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image, create its parent directory, and verify encoder success."""
    validate_image(image)
    path = Path(path).expanduser()
    ensure_output_directory(path.parent)
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as error:
        raise OSError(f"OpenCV could not save image: {path}") from error
    if not written:
        raise OSError(f"OpenCV could not save image: {path}")
    return path
