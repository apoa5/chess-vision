"""Opt-in single-photo ConvNeXt service and explicit confirmed-position boundary.

No uploaded media, draft or confirmation is retained after its request. The only
persistent process state is the frozen, reusable inference runtime.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from src.chess.board_state import BoardState
from src.chess.fen import GameState, board_from_fen, full_fen, placement_fen
from src.chess.labels import CLASSES, CLASS_NAMES
from src.chess.validation import validate_position
from src.training.multidataset import SQUARES, digest
from src.vision.convnext_runtime import CHECKPOINT_SHA, load_runtime, prepare_board_inputs
from src.vision.dataset import chess_square_at
from src.vision.geometry_audit import context_transform, project
from src.vision.position_pipeline import PreparationError, prepare_raw_photo
from scripts.run_square_evidence import make_inputs

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'configs/application/assisted_photo_v1.json'
MAX_UPLOAD = 12 * 1024 * 1024
MAX_PIXELS = 32_000_000


class PhotoError(ValueError):
    def __init__(self, code, message, *, status=422, details=None):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details or {}


def decode_photo(raw, content_type):
    """Bound allocation before decode, then explicitly apply EXIF to BGR pixels."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_UPLOAD:
        raise PhotoError('upload_size', 'Choose a JPEG or PNG photograph up to 12 MiB.', status=413)
    formats = {'image/jpeg': 'JPEG', 'image/png': 'PNG'}
    if content_type not in formats:
        raise PhotoError('unsupported_image', 'Use JPEG or PNG. Convert HEIC or other formats before uploading.', status=415)
    try:
        with Image.open(BytesIO(raw)) as header:
            # MPO uses JPEG primary-image pixels with additional MP metadata.
            # Decode only the primary image, as OpenCV already does for JPEG.
            jpeg_primary = (content_type == 'image/jpeg' and header.format == 'MPO'
                            and raw.startswith(b'\xff\xd8\xff'))
            if header.format != formats[content_type] and not jpeg_primary:
                raise PhotoError('unsupported_image',
                                 f'This file contains {header.format} data, not the declared JPEG/PNG type. Export it as JPEG or PNG before uploading.', status=415)
            width, height = header.size
            if width * height > MAX_PIXELS or min(width, height) < 64:
                raise PhotoError('image_dimensions', 'Use a photograph between 64 pixels per side and 32 megapixels.')
            orientation = header.getexif().get(274, 1)
            header.verify()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError('Decode failed')
        # Equivalent to EXIF transpose; operations preserve decoded pixel values.
        if orientation == 2: image = cv2.flip(image, 1)
        elif orientation == 3: image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 4: image = cv2.flip(image, 0)
        elif orientation == 5: image = cv2.transpose(image)
        elif orientation == 6: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 7: image = cv2.flip(cv2.transpose(image), -1)
        elif orientation == 8: image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation != 1: raise ValueError('Invalid EXIF orientation')
    except PhotoError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, cv2.error) as error:
        raise PhotoError('invalid_image', 'This photograph could not be decoded. Try a different JPEG or PNG.') from error
    return image, dict(encoded_width=width, encoded_height=height, exif_orientation=orientation,
                       width=image.shape[1], height=image.shape[0])


def manual_geometry(value, width, height):
    if not isinstance(value, dict) or set(value) - {'corners', 'white_side'}:
        raise PhotoError('invalid_geometry', 'Invalid manual alignment options.')
    side = value.get('white_side')
    if side not in (None, 'bottom', 'left', 'top', 'right'):
        raise PhotoError('invalid_geometry', 'Choose the White home edge (rank 1).')
    corners = value.get('corners')
    if corners is not None:
        try:
            points = np.asarray(corners, np.float32)
            if (points.shape != (4, 2) or not np.isfinite(points).all() or
                (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any() or
                not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < width * height * .005):
                raise ValueError('Invalid quadrilateral')
        except (ValueError, TypeError, cv2.error) as error:
            raise PhotoError('invalid_geometry', 'Select four distinct playing-area corners clockwise, starting at the image top-left.') from error
        corners = points
    return corners, side


class PhotoService:
    def __init__(self, *, reference=False):
        self.profile = json.loads(PROFILE.read_text())
        runtime_path = ROOT / self.profile['runtime_config']
        if digest(runtime_path) != self.profile['runtime_config_sha256']:
            raise ValueError('C.11 runtime configuration identity changed')
        self.runtime_config = json.loads(runtime_path.read_text())
        if self.runtime_config['checkpoint_sha256'] != CHECKPOINT_SHA:
            raise ValueError('Wrong assisted-photo checkpoint')
        if reference:
            self.runtime_config.update(backend='pytorch_reference', batch_size=16, threads=2)
        self.runtime_config['checkpoint'] = str(ROOT / self.runtime_config['checkpoint'])
        self._runtime = None
        self._load_lock = threading.Lock()

    def runtime(self):
        with self._load_lock:
            if self._runtime is None:
                self._runtime = load_runtime(self.runtime_config)
        return self._runtime

    def info(self):
        return dict(profile='assisted_convnext_photo_v1', backend=self.runtime_config['backend'],
                    checkpoint_sha256=CHECKPOINT_SHA, class_names=list(CLASS_NAMES),
                    representation=self.runtime_config['representation'], calibration='uncalibrated',
                    batch_size=self.runtime_config['batch_size'], cpu_threads=self.runtime_config['threads'],
                    loaded=self._runtime is not None, license=self.profile['license'], max_upload_bytes=MAX_UPLOAD)

    def recognize(self, raw, content_type, manual=None):
        begin = time.perf_counter(); timings = {}
        image, source = decode_photo(raw, content_type)
        timings['decode_seconds'] = time.perf_counter() - begin
        corners, side = manual_geometry(manual if manual is not None else {}, source['width'], source['height'])
        start = time.perf_counter()
        try:
            # Existing raw pipeline takes a path. A single temporary lossless PNG
            # preserves upright decoded pixels; it is removed even on failure.
            with tempfile.TemporaryDirectory(prefix='chessvision-photo-') as temporary:
                path = Path(temporary) / 'upright.png'
                if not cv2.imwrite(str(path), image): raise OSError('Temporary image encoding failed')
                rectified, grid, turns, provenance = prepare_raw_photo(
                    path, self.profile['geometry'], corners=corners, white_side=side)
        except PreparationError as error:
            messages = {
                'localization': 'Board not located. Keep the entire playing area visible, or select its four corners.',
                'orientation': 'Board orientation is uncertain. Choose the White home edge (rank 1), then retry.',
                'grid': 'Square alignment was rejected. Retake the photograph or check the selected corners.',
            }
            raise PhotoError(error.stage, messages.get(error.stage, 'Board geometry could not be resolved.'),
                             details={'image': source, 'geometry_status': 'unresolved'}) from error
        except (ValueError, cv2.error) as error:
            raise PhotoError('invalid_geometry', 'Board geometry was invalid. Check the selected playing-area corners and retry.',
                             details={'image': source, 'geometry_status': 'unresolved'}) from error
        timings['geometry_seconds'] = time.perf_counter() - start
        transform = np.linalg.inv(context_transform(provenance['source_corners'], rectified.board_bounds))
        polygons = project(grid['square_polygons'], transform)
        ids = [chess_square_at(r, c, turns) for r in range(8) for c in range(8)]
        geometry = dict(provenance=provenance, source_square_polygons=polygons.tolist(), semantic_square_ids=ids)
        start = time.perf_counter()
        try: inputs = make_inputs(image, geometry, self.profile['crop'])
        except (ValueError, cv2.error) as error:
            raise PhotoError('crop_alignment', 'Some target squares are clipped or invalid. Check alignment or retake the photograph.') from error
        by_square = {p['canonical_square']: p for p in inputs}
        packed = np.stack([by_square[s]['packed'] for s in SQUARES])
        x = prepare_board_inputs(packed)
        timings['inputs_seconds'] = time.perf_counter() - start
        start = time.perf_counter(); runtime = self.runtime()
        timings['runtime_load_seconds'] = time.perf_counter() - start
        # All HTTP CPU work is serialized by the existing server lock. Restore
        # this explicit profile's budget if a diagnostic/legacy call changed it.
        import torch
        torch.set_num_threads(self.runtime_config['threads'])
        logits = runtime.predict_logits(x)
        timings.update(runtime.last_timings)
        probabilities = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
        probabilities /= probabilities.sum(1, keepdims=True)
        predictions = logits.argmax(1)
        symbols = [CLASSES[int(i)].symbol for i in predictions]
        board = BoardState([symbols[i:i+8] for i in range(0, 64, 8)], 'white_at_bottom')
        squares = {s: dict(symbol=symbols[i], class_name=CLASS_NAMES[int(predictions[i])],
                           confidence=float(probabilities[i].max()), logits=logits[i].tolist(),
                           probabilities=probabilities[i].tolist(), visibility='unknown',
                           context_valid_fraction=by_square[s]['context_valid_fraction'])
                   for i, s in enumerate(SQUARES)}
        return dict(schema_version='assisted_photo_result_v1', recognition_id=str(uuid.uuid4()), status='draft',
                    confirmed=False, image=source, source_sha256=hashlib.sha256(raw).hexdigest(), squares=squares,
                    square_order=list(SQUARES), provisional_placement=placement_fen(board), model=self.info(),
                    geometry=dict(status='completed_unverified', owner_verified=False, policy='legacy/default',
                                  source_corners=provenance['source_corners'], rotation_quarters_ccw=int(turns),
                                  orientation_source=provenance['orientation_source'],
                                  camera_orientation=['white_near_camera','white_left','black_near_camera','white_right'][turns],
                                  square_polygons=dict(zip(ids, polygons.tolist()))),
                    packed_inputs_sha256=[hashlib.sha256(p.tobytes()).hexdigest() for p in packed],
                    timings=timings, elapsed_seconds=time.perf_counter()-begin)


def confirm_position(payload):
    """Validate explicit edits/game state; no prediction or engine is consulted."""
    if not isinstance(payload, dict) or payload.get('user_confirmed') is not True:
        raise PhotoError('confirmation_required', 'Check every square and the game-state fields, then explicitly confirm.')
    revision = payload.get('revision')
    if type(revision) is not int or not 0 <= revision <= 2**31-1:
        raise PhotoError('invalid_position', 'Invalid draft revision.')
    try:
        uuid.UUID(payload['recognition_id'])
        symbols = payload['board']
        if not isinstance(symbols, list) or len(symbols) != 64: raise ValueError('Expected 64 squares')
        board = BoardState([symbols[i:i+8] for i in range(0, 64, 8)], 'white_at_bottom')
        fields = payload['game_state']
        if not isinstance(fields, dict) or set(fields) != set(GameState.__dataclass_fields__):
            raise ValueError('Supply all five game-state fields')
        game = GameState(**fields)
        if max(game.halfmove_clock, game.fullmove_number) > 2**31-1:
            raise ValueError('Move counters are too large')
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise PhotoError('invalid_position', 'Invalid piece placement or game state: '+str(error)) from error
    validation = validate_position(board, game)
    messages = []
    for symbol, color in [('K', 'White'), ('k', 'Black')]:
        if symbols.count(symbol) != 1: messages.append(f'Place exactly one {color} king.')
    if any(p in ('p','P') for p in symbols[:8]+symbols[-8:]):
        messages.append('Pawns cannot be on rank 1 or rank 8.')
    for color, pieces, pawn in [('White','PNBRQK','P'),('Black','pnbrqk','p')]:
        if sum(p in pieces for p in symbols if p is not None) > 16: messages.append(f'{color} has more than 16 pieces.')
        if symbols.count(pawn) > 8: messages.append(f'{color} has more than 8 pawns.')
    readable = {'bad_castling_rights': 'Castling rights do not match the king/rook squares. Check the rights you supplied.',
                'invalid_ep_square': 'The en-passant square is inconsistent with the side to move or pawn placement.',
                'opposite_check': 'The side that just moved is in check. Check the pieces and side to move.',
                'too_many_checkers': 'Too many pieces are checking the king. Check the placement.',
                'impossible_check': 'The check configuration is inconsistent. Check the pieces and side to move.'}
    if not validation['standard_chess_valid']:
        for diagnostic in validation['diagnostics']:
            message = readable.get(diagnostic)
            if message: messages.append(message)
        if not messages: messages.append('This position fails standard-chess consistency checks. Check both kings and game state.')
    if messages:
        raise PhotoError('position_validation', ' '.join(dict.fromkeys(messages)),
                         details={'validation': validation, 'structural_errors': messages,
                                  'note': 'Static consistency is not a proof of legal game history.'})
    fen = full_fen(board, game)
    if placement_fen(board_from_fen(fen)) != placement_fen(board):
        raise PhotoError('invalid_position', 'FEN round-trip failed.')
    return dict(schema_version='confirmed_position_v1', status='confirmed', user_confirmed=True,
                confirmation_id=str(uuid.uuid4()), recognition_id=payload['recognition_id'], revision=revision,
                board=list(symbols), game_state=fields, placement_fen=placement_fen(board), full_fen=fen,
                validation=validation, game_state_inferred_from_image=False,
                note='User-confirmed position; static validation does not prove reachability. No engine invoked.')
