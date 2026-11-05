"""C5 spending/replay/loss-observation contracts without cloud compute."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
import pytest,torch
from src.training.chesscog_c5 import claim_once,incremental_cost,training_loss_recorder,loss_summary,prepare

def test_exclusive_claim_prevents_platform_training_replay(tmp_path):
    p=tmp_path/'claim.json';commits=[];claim_once(p,dict(run='one'),lambda:commits.append(True))
    with pytest.raises(ValueError,match='no automatic GPU retry'):claim_once(p,dict(run='two'))
    assert json.loads(p.read_text())==dict(run='one') and len(commits)==1

def test_budget_delta_uses_conservative_non_reconciled_components():
    before=dict(metered_cost='0.01',metered_cost_breakdown=dict(app='0.014'))
    after=dict(metered_cost='0.03',metered_cost_breakdown=dict(app='0.05'))
    assert incremental_cost(before,after)==pytest.approx(.036)
    assert incremental_cost(after,before)==0

def test_loss_observation_preserves_loss_gradient_and_restores_function(tmp_path):
    torch.manual_seed(42);a=torch.randn(7,13,requires_grad=True);b=a.detach().clone().requires_grad_();y=torch.tensor([0,1,2,3,4,5,6]);w=torch.arange(1.,14.)
    original=torch.nn.functional.cross_entropy;expected=original(a,y,weight=w);expected.backward()
    with training_loss_recorder(1,tmp_path) as rows:
        actual=torch.nn.functional.cross_entropy(b,y,weight=w);actual.backward()
        with torch.inference_mode():torch.nn.functional.cross_entropy(b,y,weight=w)
    assert torch.equal(expected,actual) and torch.equal(a.grad,b.grad)
    assert len(rows)==1 and torch.nn.functional.cross_entropy is original
    assert loss_summary(rows)[0]['training_loss']==pytest.approx(float(expected.detach()))

def test_loss_function_restored_after_failure(tmp_path):
    original=torch.nn.functional.cross_entropy
    with pytest.raises(RuntimeError):
        with training_loss_recorder(1,tmp_path):raise RuntimeError('interrupt')
    assert torch.nn.functional.cross_entropy is original

def test_preparation_cannot_download_external_test(tmp_path):
    with pytest.raises(ValueError,match='Only official train/validation'):
        prepare(tmp_path,dict(files={'test.zip':{}}))

def test_preparation_verifies_archives_quarantines_invalid_labels_and_preserves_roles(tmp_path,monkeypatch):
    import urllib.request,shutil
    probe=Path('outputs/evaluation/multidataset_c4/chesscog_probe')
    if not (probe/'train/0000.png').is_file():pytest.skip('C4 official sample not installed')
    root=tmp_path/'volume';root.mkdir();(root/'personal_protected_placements.json').write_text('{}')
    source=tmp_path/'source';source.mkdir();inventories={}
    for split,part,id in [('train','train.z01','0000'),('val','val.zip','0024')]:
        with zipfile.ZipFile(source/part,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for ext in ['json','png']:z.write(probe/split/(id+'.'+ext),split+'/'+id+'.'+ext)
            if split=='train':
                z.writestr('train/bad.json',json.dumps(dict(fen='invalid')));z.write(probe/split/(id+'.png'),'train/bad.png')
        with zipfile.ZipFile(source/part) as z:
            inventories[split]=dict(rows=[dict(name=i.filename,method=i.compress_type,crc32=i.CRC,compressed_bytes=i.compress_size,bytes=i.file_size,disk=0,local_offset=i.header_offset) for i in z.infolist()])
    (source/'train.zip').write_bytes(b'unused split tail fixture')
    files={n:dict(attributes=dict(size=(source/n).stat().st_size,extra=dict(hashes=dict(sha256=hashlib.sha256((source/n).read_bytes()).hexdigest()))),links=dict(download='https://fixture.invalid/'+n)) for n in ['train.z01','train.zip','val.zip']}
    plan=dict(files=files,inventories=inventories);(root/'source_plan_v1.json').write_text(json.dumps(plan))
    class Response:
        status=200
        def __init__(self,path):self.f=path.open('rb')
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        def read(self,n):return self.f.read(n)
    monkeypatch.setattr(urllib.request,'urlopen',lambda req,**kw:Response(source/req.full_url.split('/')[-1].split('?')[0]))
    monkeypatch.setattr(shutil,'disk_usage',lambda p:type('Disk',(),dict(free=100_000_000_000))())
    r=prepare(root,plan)
    assert r['official_pairs']==dict(train=2,val=1) and r['train_boards']==1 and r['val_boards']==1
    assert len(r['excluded_annotations'])==1 and r['excluded_annotations'][0]['source_role']=='train'
    assert all(x['official_sha256_verified'] for x in r['archive_receipts'])
    m=json.loads((root/'manifest.json').read_text());assert {b['split'] for b in m['boards']}=={'train','val'}
    with pytest.raises(ValueError,match='Completed preparation'):prepare(root,plan)
