"""Build board-specific coordinate templates from a clear, explicitly oriented photo."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import load_image
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.orientation import TEMPLATE_PATH, marking_patches, glyph_from_patch, rotate_rectified
from src.vision.pipeline import rectify_with_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--white-side', required=True, choices=['bottom', 'left', 'top', 'right'])
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=TEMPLATE_PATH)
    args = parser.parse_args()
    try:
        if args.output.suffix != '.npz':
            raise ValueError('Template output must have the .npz extension.')
        config = load_config(args.config)
        image = load_image(args.image)
        detected = detect_board_corners(image, DetectionSettings(**config.get('board_detection', {})))
        if not detected.success:
            raise ValueError(f'Detection failed: {detected.reason}')
        board = rotate_rectified(rectify_with_config(image, detected.corners, config),
                                 {'bottom': 0, 'left': 1, 'top': 2, 'right': 3}[args.white_side])
        bank = {}
        for edge, _, label, patch in marking_patches(board):
            if edge in ('bottom', 'left'):
                glyph = glyph_from_patch(patch)
                if glyph is not None:
                    bank[label] = glyph
        if set(bank) != set('ABCDEFGH12345678'):
            raise ValueError('All 16 bottom/left markings must be readable for calibration.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **bank)
        args.output.with_suffix('.json').write_text(json.dumps({
            'reference_image': args.image.name, 'reference_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
            'white_side': args.white_side, 'labels': list('ABCDEFGH12345678'), 'glyph_shape': [48, 36],
            'scope': 'This physical board: light printed markings; top/right characters face the opposite player.'
        }, indent=2)+'\n')
        print(f'Saved templates: {args.output}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
