"""Evaluate prepared held-out boards and all raw photographs of held-out positions."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.chess.board_state import BoardState
from src.chess.fen import GameState, board_from_fen
from src.chess.validation import fen_result
from src.utils.config import load_config
from src.utils.image_io import save_image
from src.vision.dataset import file_hash
from src.vision.inference import PieceClassifier,predict_rectified,prediction_overlay
from src.vision.position_pipeline import prepare_raw_photo,load_rectified_geometry,PreparationError
from src.vision.position_evaluation import compare_position,summarize_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--mode',choices=['both','prepared','raw'],default='both')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/evaluation/week2'))
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    cv2.setNumThreads(1)
    metadata_path = config['classifier']['dataset_paths']['metadata']
    manifest = json.loads(metadata_path.read_text()); manifest_sha = file_hash(metadata_path)
    prepared_path = metadata_path.parent/'prepared_boards.json'
    prepared = {r['image_id']:r for r in json.loads(prepared_path.read_text())}
    images_path = config['data']['metadata']/'images.json'; positions_path = config['data']['metadata']/'positions.json'
    images = json.loads(images_path.read_text()); positions = json.loads(positions_path.read_text())
    assignments = manifest['split_report']['position_assignment']
    names = sorted(name for name,entry in images.items() if assignments.get(entry['position_id'])=='test')
    crop_names = {r['source_image'] for r in manifest['results'] if r['split']=='test'}
    checkpoint_path = config['classifier']['checkpoint_path']
    checkpoint_sha = file_hash(checkpoint_path)
    classifier = PieceClassifier(checkpoint_path,batch_size=config['classifier']['batch_size'],cpu_threads=config['classifier']['cpu_threads'])
    import torch
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['manifest_sha256'] != manifest_sha or checkpoint['dataset_fingerprint'] != manifest['split_report']['dataset_fingerprint']:
        raise ValueError('Checkpoint and evaluation dataset version mismatch.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    all_rows = []; expected_cache = {pos:board_from_fen(data['fen']) for pos,data in positions.items() if assignments.get(pos)=='test'}
    for mode in (['prepared','raw'] if args.mode=='both' else [args.mode]):
        for name in names:
            if mode=='prepared' and name not in crop_names: continue
            destination = args.output_dir/mode/Path(name).stem; destination.mkdir(parents=True,exist_ok=True)
            truth = expected_cache[images[name]['position_id']]
            record = {'mode':mode,'image_id':name,'position_id':images[name]['position_id'],'status':'failed',
                      'ground_truth_placement':positions[images[name]['position_id']]['fen'].split()[0],
                      'predicted_placement':None,'square_accuracy':None,'piece_errors':None,'exact_match':False,
                      'avg_confidence':None,'inference_time_ms':None,'validation_status':'not_evaluated',
                      'failure_stage':None,'notes':'','prediction_path':None}
            start = time.perf_counter(); stage = 'preparation'
            try:
                if mode=='raw':
                    rectified,grid,turns,provenance = prepare_raw_photo(config['data']['raw']/name,config,destination)
                else:
                    cached = prepared[name]
                    if cached['fen'] != positions[images[name]['position_id']]['fen']:
                        raise ValueError('Prepared board ground truth changed since dataset creation.')
                    rectified,grid,turns,source = load_rectified_geometry(Path(cached['context_path']),prepared_path)
                    provenance = {'localization_source':cached.get('localization_source','provided_geometry'),'orientation_source':source}
                stage = 'classification'
                padding = round((rectified.board_bounds[2]-rectified.board_bounds[0])/8*config['board']['crop_padding_squares'])
                result = predict_rectified(classifier,rectified,grid,turns,padding=padding)
                board = BoardState(result['board']['rows'],'white_at_bottom')
                comparison = compare_position(board,truth)
                # Ground-truth turn/rights are deliberately not supplied to predicted diagnostics.
                result.update(fen_result(board));result.update(provenance)
                result.update(source_image=name,mode=mode,comparison=comparison)
                (destination/'predictions.json').write_text(json.dumps(result,indent=2)+'\n')
                save_image(destination/'prediction_overlay.png',prediction_overlay(rectified.image,result['predictions'],config['classifier']['confidence_threshold']))
                record.update(comparison,status='predicted',avg_confidence=float(np.mean([p['confidence'] for p in result['predictions']])),
                              inference_time_ms=result['inference_seconds']*1000,
                              validation_status='valid' if result['validation']['standard_chess_valid'] else 'diagnostics',
                              diagnostics=result['validation']['diagnostics'],rotation_quarters_ccw=turns,
                              prediction_path=str(destination/'predictions.json'),**provenance)
            except (OSError,ValueError,TypeError,KeyError,RuntimeError) as error:
                record.update(failure_stage=getattr(error,'stage',stage),notes=str(error))
            record['total_time_ms'] = (time.perf_counter()-start)*1000
            all_rows.append(record)
            (destination/'evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
            print(f"{mode} {name}: {record['piece_errors'] if record['status']=='predicted' else record['notes']}",flush=True)
    if file_hash(metadata_path) != manifest_sha or file_hash(checkpoint_path) != checkpoint_sha:
        raise ValueError('Dataset or checkpoint changed during evaluation.')
    report = {'checkpoint_sha256':checkpoint_sha,'checkpoint_path':str(checkpoint_path),'best_epoch':classifier.checkpoint_epoch,
              'manifest_sha256':manifest_sha,'dataset_fingerprint':checkpoint['dataset_fingerprint'],
              'ground_truth_positions_sha256':file_hash(positions_path),'images_metadata_sha256':file_hash(images_path),
              'test_positions':sorted(expected_cache),'modes':{mode:summarize_positions([r for r in all_rows if r['mode']==mode])
                           for mode in sorted({r['mode'] for r in all_rows})},
              'game_state_diagnostics_defaults':asdict(GameState()),'records':all_rows,
              'limitations':['Prepared evaluation uses reviewed geometry and known orientation.',
                'Raw evaluation supplies no stored corners, capture orientation, FEN or manual fallbacks to inference.',
                'Raw failures count as incorrect in exact-board accuracy over all attempts; square accuracy is successful-only.',
                'Exact FEN comparison concerns piece placement only; nonvisual game state is not evaluated.',
                'Existing holdout has three positions and has already been inspected; no newly captured images tested.',
                'Model inference latency includes preprocessing and batches but not model loading or raw localization; total time includes preparation and artifact writing.']}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (config['data']['metadata']/'day14_position_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['mode','image_id','position_id','status','ground_truth_placement','predicted_placement','square_accuracy','piece_errors','exact_match','avg_confidence','inference_time_ms','total_time_ms','validation_status','failure_stage','notes']
    with (args.output_dir/'week2_positions.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    print(json.dumps(report['modes'],indent=2))


if __name__=='__main__': main()
