"""Raw-photo preparation shared by interactive prediction and end-to-end evaluation."""
import json
from pathlib import Path
from src.vision.dataset import file_hash
import cv2
from src.utils.image_io import load_image, save_image
from src.utils.detection_debug import save_detection_debug
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.pipeline import rectify_with_config
from src.vision.perspective import order_corners, RectifiedBoard
from src.vision.orientation import resolve_orientation, OrientationSettings
from src.vision.grid_geometry import fit_grid

SIDES = {'bottom':0,'left':1,'top':2,'right':3}


class PreparationError(ValueError):
    def __init__(self,stage,message):
        super().__init__(message)
        self.stage = stage


def prepare_raw_photo(image_path,config,destination=None,*,corners=None,white_side=None):
    """No filename metadata, FEN, stored corner overrides or expected orientation."""
    cv2.setNumThreads(1)
    image = load_image(image_path)
    if corners is None:
        settings = DetectionSettings(**config.get('board_detection',{}))
        detection = detect_board_corners(image,settings)
        if destination is not None:
            save_detection_debug(destination/'detection.png',image,detection,settings)
        if not detection.success:
            raise PreparationError('localization',f'Board localization failed: {detection.reason}')
        corners = detection.corners; localization = 'automatic'
    else:
        corners = order_corners(corners); localization = 'provided_corners'
    rectified = rectify_with_config(image,corners,config)
    if destination is not None: save_image(destination/'context.png',rectified.image)
    if white_side is None:
        orientation = resolve_orientation(rectified,OrientationSettings(**config.get('orientation',{})))
        turns = orientation.rotation_quarters_ccw
        evidence = orientation.metadata(); source = 'border_markings'
    else:
        if white_side not in SIDES: raise ValueError('Unknown White side.')
        turns = SIDES[white_side]; evidence = None; source = 'provided_white_side'
    if turns is None:
        raise PreparationError('orientation','Orientation uncertain. Inspect context.png and supply --white-side bottom/left/top/right.')
    grid = fit_grid(rectified.image,rectified.board_bounds,light_top_left=turns%2==0)
    if grid['fit_status'] != 'fitted': raise PreparationError('grid','Grid fit rejected.')
    provenance = {'localization_source':localization,'orientation_source':source,
                  'orientation_evidence':evidence,'source_corners':corners.tolist()}
    return rectified,grid,turns,provenance


def load_rectified_geometry(image_path, geometry_path):
    """Load geometry only; ground-truth classes/FEN are deliberately unused."""
    geometry = json.loads(geometry_path.read_text())
    if isinstance(geometry,list):
        matches = [g for g in geometry if g.get('context_path') and Path(g['context_path']).resolve() == image_path.resolve()]
        if len(matches) != 1: raise ValueError('Geometry manifest must contain exactly one matching context_path.')
        geometry = matches[0]
    if geometry.get('context_sha256') and file_hash(image_path) != geometry['context_sha256']:
        raise ValueError('Rectified image does not match geometry hash.')
    image = load_image(image_path)
    bounds = geometry['board_bounds']
    if len(bounds) != 4 or any(type(v) is not int for v in bounds): raise ValueError('Playing bounds must be four integers.')
    x0,y0,x1,y1 = bounds
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]) or x1-x0 != y1-y0:
        raise ValueError('Playing bounds must be square and inside the context canvas.')
    mask_path = geometry.get('mask_path') or geometry.get('valid_mask_path')
    if not mask_path: raise ValueError('Geometry must supply mask_path or valid_mask_path.')
    mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]: raise ValueError('Invalid context coverage mask.')
    if geometry.get('mask_sha256') and file_hash(Path(mask_path)) != geometry['mask_sha256']:
        raise ValueError('Context mask does not match geometry hash.')
    turns = geometry.get('label_rotation',geometry.get('rotation_quarters_ccw'))
    return RectifiedBoard(image,mask,tuple(bounds)),geometry.get('grid_geometry'),turns,geometry.get('orientation_source','provided_geometry')

