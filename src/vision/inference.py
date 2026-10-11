"""Batched piece inference, camera-view geometry and orientation-aware reconstruction."""
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from src.chess.labels import CLASS_NAMES, LABEL_VERSION
from src.chess.board_state import BoardState
from src.vision.classification import SquarePrediction
from src.vision.classifier import create_classifier
from src.vision.dataset_loader import preprocess_context_crop
from src.vision.dataset import chess_square_at
from src.vision.grid_geometry import polygon_crop


class PieceClassifier:
    """Load once; predictions use only pixels and the target-square mask."""
    def __init__(self, checkpoint_path, *, batch_size=16, cpu_threads=4):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Batch size must be positive.')
        torch.set_num_threads(cpu_threads)
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if (checkpoint['architecture'] != 'mobilenet_v3_small' or checkpoint['input_channels'] != 4
                or checkpoint['label_version'] != LABEL_VERSION or checkpoint['class_names'] != list(CLASS_NAMES)):
            raise ValueError('Checkpoint architecture, input or labels are incompatible.')
        self.input_size = checkpoint['input_size']
        self.batch_size = batch_size
        self.checkpoint_epoch = checkpoint['epoch']
        self.model = create_classifier(pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, crops):
        """Crops are dictionaries with image, target_polygon, row, col."""
        if not crops:
            raise ValueError('At least one crop is required.')
        tensors = [torch.from_numpy(preprocess_context_crop(c['image'], c['target_polygon'], self.input_size)) for c in crops]
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                logits = self.model(torch.stack(tensors[start:start+self.batch_size]))
                if logits.shape != (len(tensors[start:start+self.batch_size]), 13) or not torch.isfinite(logits).all():
                    raise ValueError('Model returned invalid logits.')
                for crop, probabilities in zip(crops[start:start+self.batch_size], logits.softmax(dim=1).tolist()):
                    predictions.append(SquarePrediction.from_probabilities(crop['row'], crop['col'], probabilities))
        return predictions


def reconstruct_board(predictions, rotation_quarters_ccw):
    """Rotate output matrices to White-bottom; never rotate classifier input pixels."""
    if type(rotation_quarters_ccw) is not int or rotation_quarters_ccw not in range(4):
        raise ValueError('Resolved orientation (0..3 CCW turns) is required.')
    if len(predictions) != 64 or {(p.row,p.col) for p in predictions} != {(r,c) for r in range(8) for c in range(8)}:
        raise ValueError('Exactly one prediction per square is required.')
    symbols = np.empty((8,8), object); confidence = np.zeros((8,8))
    for p in predictions:
        symbols[p.row,p.col] = p.symbol; confidence[p.row,p.col] = p.confidence
    camera = BoardState(symbols.tolist(), 'image_relative')
    canonical = BoardState(np.rot90(symbols,rotation_quarters_ccw).tolist(), 'white_at_bottom')
    return {'camera_view':camera.metadata(), 'board':canonical.metadata(),
            'camera_confidence_matrix':confidence.tolist(),
            'confidence_matrix':np.rot90(confidence,rotation_quarters_ccw).tolist()}


def predict_rectified(classifier, rectified, grid_geometry, rotation_quarters_ccw, *, padding=200):
    if grid_geometry.get('fit_status') != 'fitted':
        raise ValueError('Grid fit rejected; correct playing-area geometry before inference.')
    polygons = np.asarray(grid_geometry['square_polygons'],np.float32)
    if polygons.shape != (64,4,2) or not np.isfinite(polygons).all():
        raise ValueError('Expected 64 finite square polygons.')
    crops = []
    for index, polygon in enumerate(polygons):
        crop, coverage, _, bounds, local_polygon, valid = polygon_crop(rectified.image,rectified.valid_mask,polygon,padding)
        if valid < .99:
            raise ValueError(f'Square r{index//8} c{index%8} is clipped by the source image.')
        crops.append({'image':crop,'target_polygon':local_polygon,'row':index//8,'col':index%8,
                      'context_bounds':bounds,'target_valid_fraction':valid,
                      'context_valid_fraction':float((coverage>0).mean())})
    start = time.perf_counter(); predictions = classifier.predict(crops); elapsed = time.perf_counter()-start
    result = reconstruct_board(predictions,rotation_quarters_ccw)
    result.update(predictions=[{**p.metadata(),'chess_square':chess_square_at(p.row,p.col,rotation_quarters_ccw),
                               'square_polygon':polygons[i].tolist(),
                               **{key:crops[i][key] for key in ('context_bounds','target_polygon','target_valid_fraction','context_valid_fraction')}}
                              for i,p in enumerate(predictions)],
                  rotation_quarters_ccw=rotation_quarters_ccw, inference_seconds=elapsed,
                  checkpoint_epoch=classifier.checkpoint_epoch)
    return result


def prediction_overlay(image, predictions, threshold=.7):
    """Draw camera-view polygons, chess coordinates, symbols and confidence."""
    overlay = image.copy()
    for p in predictions:
        polygon = np.rint(p['square_polygon']).astype(np.int32)
        color = (0,180,0) if p['confidence'] >= threshold else (0,130,255)
        cv2.polylines(overlay,[polygon],True,color,2,cv2.LINE_AA)
        centre = polygon.mean(axis=0).astype(int)
        text = f"{p['chess_square']} {p['symbol'] or '.'} {p['confidence']:.2f}"
        scale = max(.30,min(.48,np.linalg.norm(polygon[1]-polygon[0])/220))
        size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0]
        x,y = int(centre[0]-size[0]/2),int(centre[1]+size[1]/2)
        cv2.rectangle(overlay,(x-2,y-size[1]-3),(x+size[0]+2,y+3),(245,245,245),-1)
        cv2.putText(overlay,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(15,15,15),1,cv2.LINE_AA)
    return overlay
