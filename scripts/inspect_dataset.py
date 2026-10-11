"""Create reproducible label review sheets; annotations never enter training crops."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.utils.image_io import load_image, save_image
from src.chess.labels import CLASSES


def sheet(items, path, columns=2, size=500):
    canvas = np.full((((len(items)+columns-1)//columns)*(size+45), columns*size, 3), 245, np.uint8)
    for i, (picture, title) in enumerate(items):
        y, x = divmod(i, columns)
        top, left = y*(size+45), x*size
        canvas[top:top+size, left:left+size] = cv2.resize(picture, (size,size), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, title, (left+8, top+size+28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 1, cv2.LINE_AA)
    save_image(path, canvas)
    if Path(path).suffix.lower() != '.png':
        save_image(Path(path).with_suffix('.png'), canvas)


def inspect(output, destination, boards=False, count=5, seed=42, reviewed_only=False, candidates=False):
    destination.mkdir(parents=True, exist_ok=True)
    if boards:
        records = json.loads((output/'prepared_boards.json').read_text())
        valid = [r for r in records if r.get('context_path')]
        for position in sorted({r['position_id'] for r in valid}):
            subset = [r for r in valid if r['position_id']==position]
            items = []
            for r in subset:
                picture = load_image(Path(r['context_path']).with_name(Path(r['image_id']).stem+'_labels.png'))
                save_image(destination/'boards'/f"{Path(r['image_id']).stem}.png", picture)
                items.append((picture, r['image_id']))
            sheet(items, destination/f'{position}.jpg', size=items[0][0].shape[0])
        return
    manifest = json.loads((output/('candidate_crops.json' if candidates else 'crops.json')).read_text())
    rng = np.random.default_rng(seed)
    selected = []
    for label in CLASSES:
        rows = [r for r in manifest['results'] if r['class_id']==label.class_id
                and (not reviewed_only or r.get('visual_review_approved'))]
        if not rows:
            continue
        indices = rng.choice(len(rows), min(count,len(rows)), replace=False)
        examples = []
        for index in indices:
            row = rows[int(index)]
            picture = load_image(row['crop_path'])
            x0,y0,x1,y1 = row['target_bounds']
            cv2.polylines(picture,[np.rint(row['target_polygon']).astype(np.int32)],True,(0,255,0),2) if row.get('target_polygon') else cv2.rectangle(picture, (x0,y0), (x1-1,y1-1), (0,255,0), 2)
            examples.append((picture, f"{row['source_image']} {row['chess_square']}"))
            selected.append(row['crop_id'])
        sheet(examples, destination/f'{label.name}.jpg')
    (destination/'sample_ids.json').write_text(json.dumps({'seed':seed,'crop_ids':selected},indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/classification'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/week1/day9_review'))
    parser.add_argument('--candidates', action='store_true', help='Inspect unapproved polygon crop candidates')
    parser.add_argument('--boards', action='store_true')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--reviewed-only', action='store_true', help='Show individually approved visibility samples')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    inspect(args.dataset, args.output_dir, args.boards, args.count, reviewed_only=args.reviewed_only, candidates=args.candidates)

if __name__=='__main__':
    main()
