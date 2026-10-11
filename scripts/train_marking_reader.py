#!/usr/bin/env python3
"""Train the board-font reader from reviewed, hash-bound geometry (never FEN)."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from src.scanning.marking_model import build_model,LABELS,MODEL_PATH,components
from src.vision.orientation import marking_patches,glyph_from_patch,rotate_rectified,load_templates
from src.vision.perspective import RectifiedBoard


def main():
    torch.set_num_threads(4);torch.manual_seed(42);cv2.setNumThreads(1);rng=np.random.default_rng(42)
    if '--from-prepared' in sys.argv:
        out=Path('outputs/scanning/marking_training');data=np.load(out/'samples.npz',allow_pickle=False)
        samples=[(glyph,int(label),'validation' if val else 'train','prepared') for glyph,label,val in zip(data['images'],data['labels'],data['validation'])]
    else:
        records=json.loads(Path('data/classification/prepared_boards.json').read_text());reviews=json.loads(Path('data/metadata/day9_board_reviews.json').read_text())
        validation={'position_003','position_007','position_011'};excluded={'position_006','position_010','position_016'};samples=[];templates=load_templates();discarded=0
        for record in records:
            if not record.get('context_path') or record['position_id'] in excluded:continue
            review=reviews.get(record['image_id'],{})
            if not review.get('alignment_correct') or not review.get('labels_correct') or review['signature']!=record['review_signature']:continue
            path=Path(record['context_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['context_sha256']:raise ValueError('Stale reviewed board')
            image=cv2.imread(str(path));mask=cv2.imread(record['mask_path'],0)
            board=rotate_rectified(RectifiedBoard(image,mask,tuple(record['board_bounds'])),record['label_rotation'])
            split='validation' if record['position_id'] in validation else 'train'
            for edge,index,label,patch in marking_patches(board):
                glyph=glyph_from_patch(patch)
                if glyph is not None:
                    scores={name:float(cv2.matchTemplate(glyph,template,cv2.TM_CCOEFF_NORMED)[0,0]) for name,template in templates.items()}
                    if max(scores,key=scores.get)==label and scores[label]>=.65:samples.append((glyph,LABELS.index(label),split,record['image_id']))
                    else:discarded+=1
            x0,y0,x1,y1=board.board_bounds
            negatives=components(board.image[y0:y1,x0:x1])
            for p in negatives[:24]:samples.append((p['glyph'],16,split,record['image_id']))
        out=Path('outputs/scanning/marking_training');out.mkdir(parents=True,exist_ok=True)
        # Human-readable examples: inferred labels need inspection before trusting training.
        tiles=[]
        for label in range(17):
            options=[s for s in samples if s[1]==label and s[2]=='train']
            for glyph,_,_,_ in options[:8]:
                tile=np.zeros((80,72,3),np.uint8);tile[:48,18:54]=cv2.cvtColor(glyph,cv2.COLOR_GRAY2BGR);cv2.putText(tile,LABELS[label],(5,70),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);tiles.append(tile)
        while len(tiles)%8:tiles.append(np.zeros((80,72,3),np.uint8))
        cv2.imwrite(str(out/'label_examples.jpg'),np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]))
        np.savez_compressed(out/'samples.npz',images=np.stack([s[0] for s in samples]),labels=np.array([s[1] for s in samples]),validation=np.array([s[2]=='validation' for s in samples]))
        manifest={'schema_version':'board_marking_training_v1','labels':list(LABELS),'sources':sorted(set(s[3] for s in samples)),'source_hashes':{r['image_id']:r['context_sha256'] for r in records if r['image_id'] in {s[3] for s in samples}},'validation_positions':sorted(validation),'excluded_positions':sorted(excluded),'discarded_ambiguous_glyphs':discarded,'quality_filter':'expected label agrees with template; NCC >= .65','sample_count':len(samples),'fen_used':False,'phone_scans_used_for_training':False,'label_source':'character slots from user-reviewed board geometry; glyphs are automatically isolated'}
        (out/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Dataset ready',len(samples),'examples; preview',out/'label_examples.jpg',flush=True)
        if '--prepare-only' in sys.argv:return
    train=[s for s in samples if s[2]=='train'];val=[s for s in samples if s[2]=='validation'];model=build_model();optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=-1.;history=[];start=time.perf_counter()
    weights=np.bincount([s[1] for s in train],minlength=17);weights=1/np.sqrt(np.maximum(weights,1));weights=weights/weights.mean();loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    def augment(glyph):
        angle=rng.uniform(-22,22);scale=rng.uniform(.8,1.15);h=cv2.getRotationMatrix2D((18,24),angle,scale);h[:,2]+=rng.uniform(-3,3,2);g=cv2.warpAffine(glyph,h,(36,48))
        sx,sy=rng.uniform(.65,1.25,2);shear=rng.uniform(-.22,.22)
        affine=np.float32([[sx,shear,18-18*sx-24*shear],[rng.uniform(-.15,.15),sy,24-24*sy]])
        g=cv2.warpAffine(g,affine,(36,48))
        y,x=np.where(g>30)
        if len(x):
            from src.scanning.marking_model import normalize_component
            g=normalize_component(g[y.min():y.max()+1,x.min():x.max()+1])
        if rng.random()<.3:g=cv2.GaussianBlur(g,(3,3),rng.uniform(.3,1.2))
        return np.clip(g.astype(np.float32)/255+rng.normal(0,.035,g.shape),0,1).astype(np.float32)
    for epoch in range(25):
        model.train();order=rng.permutation(len(train));losses=[]
        for i in range(0,len(order),128):
            batch=[train[j] for j in order[i:i+128]];x=torch.from_numpy(np.stack([augment(s[0]) for s in batch])[:,None]);y=torch.tensor([s[1] for s in batch]);optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval();pred=[]
        with torch.inference_mode():
            for i in range(0,len(val),128):pred.extend(model(torch.from_numpy(np.stack([s[0] for s in val[i:i+128]])[:,None].astype(np.float32)/255)).argmax(1).tolist())
        truth=np.array([s[1] for s in val]);pred=np.array(pred);accuracy=float(np.mean(pred==truth));macro=float(np.mean([np.mean(pred[truth==label]==label) for label in range(17)]));history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_accuracy':accuracy,'validation_macro_recall':macro});print(history[-1],flush=True)
        if macro>best:
            best=macro;MODEL_PATH.parent.mkdir(parents=True,exist_ok=True);torch.save({'schema_version':'board_marking_reader_v1','labels':list(LABELS),'state_dict':model.state_dict(),'epoch':epoch+1,'validation_macro_recall':macro,'dataset_sha256':hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest()},MODEL_PATH)
    (out/'training.json').write_text(json.dumps({'history':history,'elapsed_seconds':time.perf_counter()-start,'best_validation_macro_recall':best,'checkpoint_sha256':hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':main()
