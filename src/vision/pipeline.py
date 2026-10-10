"""Reusable Week 1 orchestration, with explicit failures and inspectable artifacts."""

import hashlib
import json
from pathlib import Path

from src.utils.detection_debug import save_detection_debug
from src.utils.image_io import ensure_output_directory, load_image, save_image
from src.utils.visualization import draw_board_grid
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.evaluation import detection_signature
from src.vision.perspective import warp_board_with_context
from src.vision.square_extraction import extract_squares, extract_context_squares


def save_square_crops(squares, directory):
    """Save lossless crops and their exclusive bounds in deterministic order."""
    directory = ensure_output_directory(directory)
    metadata = []
    for square in squares:
        filename = f"r{square.row}_c{square.col}.png"
        save_image(directory / filename, square.image)
        metadata.append({**square.metadata(), "filename": filename})
    (directory / "squares.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return directory


def rectify_with_config(image, corners, config):
    size = config["board"]["normalized_size"]
    margin = round(size / 8 * config["board"].get("context_margin_squares", 2.0))
    return warp_board_with_context(image, corners, size, margin)


def save_rectified_context(rectified, board_path):
    """Save the context canvas, coverage mask, and geometry alongside the core."""
    board_path = Path(board_path)
    context_path = board_path.with_name(f"{board_path.stem}_context.png")
    mask_path = board_path.with_name(f"{board_path.stem}_context_valid.png")
    grid_path = board_path.with_name(f"{board_path.stem}_context_grid.png")
    save_image(context_path, rectified.image)
    save_image(mask_path, rectified.valid_mask)
    grid = rectified.image.copy()
    x0, y0, x1, y1 = rectified.board_bounds
    grid[y0:y1, x0:x1] = draw_board_grid(rectified.board_image)
    save_image(grid_path, grid)
    metadata = {**rectified.metadata(), "context_path": str(context_path),
                "valid_mask_path": str(mask_path), "context_grid_path": str(grid_path)}
    board_path.with_name(f"{board_path.stem}_context.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def save_context_crops(rectified, directory, padding_pixels):
    directory = ensure_output_directory(directory)
    metadata = []
    side = rectified.board_image.shape[0] // 8
    for square, mask in extract_context_squares(rectified, padding_pixels):
        filename = f"r{square.row}_c{square.col}.png"
        mask_name = f"r{square.row}_c{square.col}_valid.png"
        save_image(directory / filename, square.image)
        save_image(directory / mask_name, mask)
        metadata.append({**square.metadata(), "filename": filename, "valid_mask_filename": mask_name,
                         "target_bounds": [padding_pixels, padding_pixels, padding_pixels + side, padding_pixels + side],
                         "bounds_coordinate_system": "context_canvas", "valid_fraction": float((mask > 0).mean())})
    (directory / "squares.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return directory


def process_board_image(image_path, config, output_dir=None):
    """Detect once, warp original pixels once, and extract unannotated crops.

    Failed localization still saves detection debug artifacts, but returns no
    current-run board/crop paths. Existing artifacts from earlier runs are kept.
    Review signatures bind the source, detector result, and actual warped pixels.
    """
    source = Path(image_path)
    image = load_image(source)
    settings = DetectionSettings(**config.get("board_detection", {}))
    detection = detect_board_corners(image, settings)
    paths = config["outputs"] if output_dir is None else {
        key: Path(output_dir) / key for key in ("board_detection", "rectified_boards", "square_crops")}
    debug = save_detection_debug(paths["board_detection"] / f"{source.stem}_corners.png", image, detection, settings)
    signature = detection_signature(source, detection, settings)
    record = {"image_id": source.name, "board_detected": detection.success,
              "reason": detection.reason, "candidate_score": detection.score,
              "corners": debug["corners"], "detection_signature": signature,
              "overlay_path": debug["overlay_path"], "board_path": None, "grid_path": None,
              "crop_directory": None, "square_count": 0, "output_shape": None,
              "context_path": None, "context_grid_path": None, "valid_mask_path": None,
              "context_crop_directory": None, "context_square_count": 0,
              "warp_signature": None}
    if detection.success:
        rectified = rectify_with_config(image, detection.corners, config)
        board = rectified.board_image
        squares = extract_squares(board)
        board_path = save_image(paths["rectified_boards"] / f"{source.stem}.png", board)
        grid_path = save_image(paths["rectified_boards"] / f"{source.stem}_grid.png", draw_board_grid(board))
        crops = save_square_crops(squares, paths["square_crops"] / source.stem)
        context = save_rectified_context(rectified, board_path)
        padding = round(board.shape[0] / 8 * config["board"].get("crop_padding_squares", 2.0))
        context_crops = save_context_crops(rectified, crops / "context", padding)
        warp_signature = hashlib.sha256(signature.encode() + str(board.shape).encode() + board.tobytes()).hexdigest()
        record.update(board_path=str(board_path), grid_path=str(grid_path), crop_directory=str(crops),
                      square_count=len(squares), output_shape=list(board.shape), warp_signature=warp_signature,
                      **context, context_crop_directory=str(context_crops), context_square_count=64,
                      crop_padding_pixels=padding)
    report_path = Path(debug["overlay_path"]).with_name(f"{source.stem}_pipeline.json")
    report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
