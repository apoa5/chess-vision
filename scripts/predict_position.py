"""Predict all 64 squares from a raw photo or a rectified context canvas."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.chess.board_state import BoardState
from src.chess.fen import GameState
from src.chess.validation import fen_result
from src.vision.position_pipeline import prepare_raw_photo, load_rectified_geometry
from src.utils.config import load_config
from src.utils.image_io import load_image, save_image
from src.vision.perspective import RectifiedBoard
from src.vision.orientation import resolve_orientation, OrientationSettings
from src.vision.grid_geometry import fit_grid
from src.vision.dataset import file_hash
from src.vision.inference import PieceClassifier, predict_rectified, prediction_overlay

SIDES = {'bottom':0,'left':1,'top':2,'right':3}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--mode',choices=['raw','rectified'],default='raw')
    parser.add_argument('--geometry',type=Path,help='Rectified geometry JSON or prepared_boards.json; never uses its FEN/labels.')
    parser.add_argument('--corners',type=Path,help='Explicit raw-image playing-area corners JSON: {"corners": [[x,y], ...]}.')
    parser.add_argument('--white-side',choices=list(SIDES),help='White side in camera-view rectified image, if markings are unreadable.')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/week2/predictions'))
    parser.add_argument('--side-to-move',choices=['w','b'],default=None)
    parser.add_argument('--castling',default=None,help='Explicit rights, e.g. KQkq; defaults to -.')
    parser.add_argument('--en-passant',default=None,help='Explicit target square; defaults to -.')
    parser.add_argument('--halfmove-clock',type=int,default=None)
    parser.add_argument('--fullmove-number',type=int,default=None)
    args = parser.parse_args()
    destination = args.output_dir / args.image.stem
    destination.mkdir(parents=True,exist_ok=True)
    try:
        supplied = {name:getattr(args,name) for name in GameState.__dataclass_fields__ if getattr(args,name) is not None}
        game_state = GameState(**supplied)
        config = load_config(args.config) if args.config else load_config()
        cv2.setNumThreads(1)
        if args.mode == 'raw':
            if args.geometry: raise ValueError('--geometry is for rectified input only.')
            provided_corners = json.loads(args.corners.read_text())['corners'] if args.corners else None
            rectified,grid,turns,provenance = prepare_raw_photo(args.image,config,destination,
                                                              corners=provided_corners,white_side=args.white_side)
            localization = provenance['localization_source']
            orientation_source = provenance['orientation_source']
        else:
            if args.corners: raise ValueError('--corners is for raw input only.')
            if not args.geometry: raise ValueError('Rectified input requires --geometry with playing bounds and coverage mask.')
            rectified,grid,turns,orientation_source = load_rectified_geometry(args.image,args.geometry)
            localization = 'provided_rectified_geometry'
        orientation_evidence = provenance['orientation_evidence'] if args.mode == 'raw' else None
        if args.white_side:
            turns = SIDES[args.white_side]; orientation_source = 'provided_white_side'
        elif turns is None:
            orientation = resolve_orientation(rectified,OrientationSettings(**config.get('orientation',{})))
            orientation_evidence = orientation.metadata()
            turns = orientation.rotation_quarters_ccw; orientation_source = 'border_markings'
        if turns is None:
            save_image(destination/'context.png',rectified.image)
            raise ValueError('Orientation uncertain. Inspect context.png and supply --white-side bottom/left/top/right.')
        if type(turns) is not int or turns not in range(4): raise ValueError('Invalid orientation rotation.')
        if grid is None:
            grid = fit_grid(rectified.image,rectified.board_bounds,light_top_left=turns%2==0)
        classifier = PieceClassifier(config['classifier']['checkpoint_path'],batch_size=config['classifier']['batch_size'],
                                     cpu_threads=config['classifier']['cpu_threads'])
        padding = round((rectified.board_bounds[2]-rectified.board_bounds[0])/8 * config['board']['crop_padding_squares'])
        result = predict_rectified(classifier,rectified,grid,turns,padding=padding)
        result.update(fen_result(BoardState(result['board']['rows'],'white_at_bottom'),game_state,supplied))
        if args.mode == 'raw': result['source_corners'] = provenance['source_corners']
        result.update(source_image=str(args.image.resolve()),source_sha256=file_hash(args.image),
                      checkpoint_path=str(config['classifier']['checkpoint_path']),
                      checkpoint_sha256=file_hash(config['classifier']['checkpoint_path']),
                      localization_source=localization,orientation_source=orientation_source,
                      orientation_evidence=orientation_evidence,board_bounds=list(rectified.board_bounds),
                      grid_geometry=grid,geometry_review_required=grid.get('review_required',True),
                      confidence_threshold=config['classifier']['confidence_threshold'],
                      low_confidence_squares=[p['chess_square'] for p in result['predictions']
                                             if p['confidence'] < config['classifier']['confidence_threshold']],
                      class_names=list(CLASS_NAMES),
                      status='predicted',warnings=['Predictions are uncorrected model outputs; confidence does not guarantee correctness.',
                           'Inspect grid alignment; automatic geometry has not been manually approved for this run.']
                      if localization == 'automatic' else ['Predictions are uncorrected model outputs; confidence does not guarantee correctness.'])
        save_image(destination/'context.png',rectified.image)
        save_image(destination/'context_valid.png',rectified.valid_mask)
        save_image(destination/'prediction_overlay.png',prediction_overlay(rectified.image,result['predictions'],config['classifier']['confidence_threshold']))
        geometry_export = {'board_bounds':list(rectified.board_bounds),'mask_path':str((destination/'context_valid.png').resolve()),
                           'context_sha256':file_hash(destination/'context.png'),'mask_sha256':file_hash(destination/'context_valid.png'),
                           'rotation_quarters_ccw':turns,'orientation_source':orientation_source,'grid_geometry':grid}
        (destination/'geometry.json').write_text(json.dumps(geometry_export,indent=2)+'\n')
        (destination/'predictions.json').write_text(json.dumps(result,indent=2)+'\n')
        (destination/'status.json').write_text(json.dumps({'status':'predicted','predictions_path':str(destination/'predictions.json')},indent=2)+'\n')
        print('Predicted board (White at bottom; ranks 8 to 1):')
        for row in result['board']['rows']: print(' '.join(symbol or '.' for symbol in row))
        print(f"Piece-placement FEN: {result['placement_fen']}")
        print(f"Full FEN (game state supplied/defaulted): {result['full_fen']}")
        print(f"Validation: {'standard-chess consistent' if result['validation']['standard_chess_valid'] else ', '.join(result['validation']['diagnostics'])}")
        print(f"64 predictions saved: {destination/'predictions.json'}")
        print(f"Overlay: {destination/'prediction_overlay.png'}")
        print(f"Low-confidence squares: {', '.join(result['low_confidence_squares']) or 'none'}")
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as error:
        (destination/'status.json').write_text(json.dumps({'status':'failed','error':str(error)},indent=2)+'\n')
        parser.exit(1,f'Error: {error}\n')


if __name__ == '__main__': main()
