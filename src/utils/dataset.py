"""Validate the real-photo dataset and its fixed development manifest."""

from collections import Counter
import json
from pathlib import Path
import re

import chess

from src.utils.config import PROJECT_ROOT
from src.utils.image_io import load_image


def _read_mapping(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def validate_dataset(metadata_dir: str | Path = PROJECT_ROOT / "data/metadata",
                     raw_dir: str | Path = PROJECT_ROOT / "data/raw") -> dict:
    """Check coverage, FENs, links, capture fields, decoded images, and subset.

    Return a JSON-compatible summary. A valid FEN does not establish that the
    physical pieces match it; that remains the photographer's ground truth.
    The original collection target is reported separately from validation.
    """
    metadata_dir, raw_dir = Path(metadata_dir), Path(raw_dir)
    positions = _read_mapping(metadata_dir / "positions.json")
    images = _read_mapping(metadata_dir / "images.json")
    variations = _read_mapping(metadata_dir / "variations.json")
    manifest = _read_mapping(metadata_dir / "development_subset.json")
    setup = _read_mapping(metadata_dir / "capture_setup.json")
    if not images or not positions:
        raise ValueError("The dataset must contain images and positions.")
    files = {p.name for p in raw_dir.iterdir() if p.is_file() and p.name != ".gitkeep"}
    if files != set(images):
        raise ValueError(f"Raw/metadata mismatch: missing metadata {sorted(files - set(images))}; "
                         f"missing files {sorted(set(images) - files)}")
    for position_id, position in positions.items():
        fen = position.get("fen")
        if not isinstance(fen, str) or len(fen.split()) != 6:
            raise ValueError(f"{position_id}: FEN must contain all six fields.")
        try:
            board = chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"{position_id}: invalid FEN: {error}") from error
        if not board.is_valid():
            raise ValueError(f"{position_id}: invalid chess position (status {board.status()}).")

    sizes = Counter()
    orientations_unknown = []
    for filename, entry in images.items():
        match = re.fullmatch(r"(position_\d{3})_(var_\d{2})\.(jpg|jpeg|png)", filename)
        if match is None:
            raise ValueError(f"Invalid dataset filename: {filename}")
        position_id, variation_id, _ = match.groups()
        if entry.get("position_id") != position_id or position_id not in positions:
            raise ValueError(f"{filename}: invalid position link.")
        if entry.get("variation_id") != variation_id or variation_id not in variations:
            raise ValueError(f"{filename}: invalid variation link.")
        expected_path = raw_dir / filename
        if (PROJECT_ROOT / entry.get("path", "")).resolve() != expected_path.resolve():
            raise ValueError(f"{filename}: image path does not match its raw file.")
        for field in ("camera_angle_degrees", "environment", "lighting"):
            if entry.get(field) != variations[variation_id].get(field):
                raise ValueError(f"{filename}: {field} disagrees with its variation definition.")
        orientation = entry.get("orientation")
        if orientation not in (None, "white_near_camera", "black_near_camera", "white_left", "white_right"):
            raise ValueError(f"{filename}: unknown orientation value.")
        if orientation is None:
            orientations_unknown.append(filename)
        if type(entry.get("development_subset")) is not bool:
            raise ValueError(f"{filename}: development_subset must be a boolean.")
        image = load_image(expected_path)
        height, width = image.shape[:2]
        sizes[f"{width}x{height}"] += 1

    selected = manifest.get("image_ids")
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
        raise ValueError("Development manifest must contain an image_ids list of filenames.")
    if len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError("Fixed development subset must contain exactly 20 unique images.")
    flagged = {name for name, entry in images.items() if entry["development_subset"]}
    if set(selected) != flagged:
        raise ValueError("Development manifest does not match image membership flags.")
    if not set(selected).issubset(images):
        raise ValueError("Development manifest contains unknown images.")
    unused_positions = sorted(set(positions) - {e["position_id"] for e in images.values()})
    if unused_positions:
        raise ValueError(f"Positions without images: {unused_positions}")

    def counts(entries, field):
        return dict(sorted(Counter(entry[field] for entry in entries).items()))

    plan = setup["capture_plan"]
    return {
        "image_count": len(images),
        "position_count": len(positions),
        "valid_fen_count": len(positions),
        "decoded_image_count": len(images),
        "image_dimensions": dict(sorted(sizes.items())),
        "images_by_variation": counts(images.values(), "variation_id"),
        "images_by_environment": counts(images.values(), "environment"),
        "unknown_orientation_images": orientations_unknown,
        "development_subset": {
            "image_count": len(selected),
            "position_count": len({images[name]["position_id"] for name in selected}),
            "images_by_variation": counts([images[name] for name in selected], "variation_id"),
            "images_by_environment": counts([images[name] for name in selected], "environment"),
        },
        "collection_target": {
            "positions": plan["target_position_count"],
            "images": plan["target_position_count"] * plan["target_variations_per_position"],
            "missing_variations_by_position": {
                position_id: [v for v in variations if not any(
                    e["position_id"] == position_id and e["variation_id"] == v for e in images.values())]
                for position_id in sorted(positions)
            },
        },
        "limitations": [
            "FEN validity is checked; correspondence to photographed pieces is not automatically verified.",
            "Camera angles and lighting are user-supplied capture metadata, not measured from pixels.",
            "Subset is for development, not an independent held-out test set.",
        ],
    }
