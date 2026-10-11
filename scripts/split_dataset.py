"""Create position-grouped splits, class counts and training-only loss weights."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src.vision.dataset_splits import split_dataset,SPLITS
from src.chess.labels import CLASSES
from src.vision.orientation import TEMPLATE_PATH


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    config=load_config(args.config) if args.config else load_config()
    settings=config['classifier'];path=settings['dataset_paths']['metadata']
    manifest=json.loads(path.read_text())
    positions=json.loads((config['data']['metadata']/'positions.json').read_text())
    boards=json.loads((path.parent/'prepared_boards.json').read_text())
    train_only=[]
    template_metadata=TEMPLATE_PATH.with_suffix('.json')
    if template_metadata.exists():
        reference=json.loads(template_metadata.read_text()).get('reference_image')
        images=json.loads((config['data']['metadata']/'images.json').read_text())
        if reference in images:
            position=images[reference]['position_id']
            if position in {r['position_id'] for r in manifest['results']}:
                train_only.append(position)
    result,report=split_dataset(manifest,seed=settings['seed'],fractions=settings['split_fractions'],
                                position_fens={k:v['fen'] for k,v in positions.items()},source_hashes={r['image_id']:r['source_sha256'] for r in boards},trials=settings['split_search_trials'],train_only_positions=train_only)
    path.write_text(json.dumps(result,indent=2)+'\n')
    for split in SPLITS:
        destination=settings['dataset_paths'][split];destination.mkdir(parents=True,exist_ok=True)
        subset={**result,'results':[r for r in result['results'] if r['split']==split]}
        subset.update(crop_count=len(subset['results']),board_count=len(subset['results'])//64,
                      class_counts=report['splits'][split]['class_counts'])
        (destination/'manifest.json').write_text(json.dumps(subset,indent=2)+'\n')
    metadata=config['data']['metadata']
    (metadata/'day10_split_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (metadata/'day10_class_distribution.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['class_id','class_name',*SPLITS,'training_loss_weight'])
        for label in CLASSES:
            writer.writerow([label.class_id,label.name,*[report['splits'][s]['class_counts'][label.name] for s in SPLITS],report['class_weights'][label.name]])
    summary_path=path.parent/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        summary.update(split_status=result['split_status'],training_ready=result['training_ready'],
                       split_counts={s:report['splits'][s]['crop_count'] for s in SPLITS})
        summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    for split,part in report['splits'].items():
        print(f"{split}: {len(part['positions'])} positions, {part['source_image_count']} boards, {part['crop_count']} crops")
    print('Missing classes:',report['missing_classes'])

if __name__=='__main__':
    main()
