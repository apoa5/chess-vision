"""C9 source separation, masks, architecture, and coverage rather than replacement."""
import copy
import pytest
import torch
from src.training.dataset_c9 import coverage_batches,audit_splits,connected_groups
from src.training.multidataset import regular_observations

def board(i,source='chessvision_photos',fen='8/8/8/8/8/8/8/4K2k',split='train'):
 return dict(id=str(i),dataset=source,position_id=str(i),placement=fen,session_id=None,source=dict(sha256=str(i)*64),split=split,observations=regular_observations(fen,0))

def test_exact_placement_ignores_game_state_and_links_sources():
 a=board(1);b=board(2,'a3_pilot');b['placement']+=' b - - 0 12'
 assert len(connected_groups([a,b]))==1
 b['split']='val'
 with pytest.raises(ValueError):audit_splits([a,b])

def test_session_and_shared_cohort_connected():
 a=board(1);b=board(2,fen='8/8/8/8/8/8/P7/4K2k');a['independence_group_id']=b['independence_group_id']='shared'
 assert len(connected_groups([a,b]))==1

@pytest.mark.parametrize('size',[8,16,32,64])
def test_source_coverage_never_replaces_photos(size):
 class Toy:pass
 ds=Toy();ds.boards=[board(1),board(2,'d3c_retired',fen='8/8/8/8/8/8/P7/4K2k')];ds.rows=[(b,o) for b in ds.boards for o in b['observations']]
 batches=coverage_batches(ds,size,42,0);flat=sum(batches,[])
 assert sorted(flat)==list(range(128)) and len(flat)==len(set(flat))
 assert batches==coverage_batches(ds,size,42,0)

def test_four_channel_stem_zero_preserves_rgb_and_13_classes():
 pytest.importorskip('timm')
 from src.training.convnext_c9 import make_model,phase_parameters,set_phase
 torch.set_num_threads(2);model=make_model().eval()
 assert model.stem[0].weight.shape==(96,4,4,4)
 assert torch.count_nonzero(model.stem[0].weight[:,3:])==0
 x=torch.randn(1,4,224,224);other=x.clone();other[:,3]=1
 with torch.inference_mode():
  a=model(x);b=model(other)
 assert a.shape==(1,13) and torch.equal(a,b)
 c=dict(head_stem_lr=dict(stem=.001,head=.001),full_lr=dict(stem=.0001,head=.0001,backbone=.00001),weight_decay=.05)
 groups=phase_parameters(model,'head_stem',c);set_phase(model,'head_stem')
 assert all(not p.requires_grad for p in model.stages.parameters())
 model(x).sum().backward();assert model.stem[0].weight.grad is not None
 assert model.head.fc.weight.grad is not None

def test_video_coverage_preserves_whole_photo_passes():
 class Toy:pass
 ds=Toy();ds.boards=[board(1),board(2,'a3_pilot',fen='8/8/8/8/8/8/P7/4K2k'),board(3,'d3c_retired',fen='8/8/8/8/8/8/1P6/4K2k')];ds.rows=[(b,o) for b in ds.boards for o in b['observations']]
 flat=sum(coverage_batches(ds,32,42,0),[]);counts={}
 for i in flat:
  b,o=ds.rows[i%len(ds.rows)];counts[b['dataset']]=counts.get(b['dataset'],0)+1
 assert counts['chessvision_photos']==128 and counts['a3_pilot']==counts['d3c_retired']==64
 assert {i%len(ds.rows) for i in flat}==set(range(192))

def test_cost_and_resource_guards():
 from src.training.trainer_c9 import settings_guard
 from src.training.convnext_c9 import MODEL_NAME
 c=dict(architecture=MODEL_NAME,representation='b2-context1-224-v1',seed=42,budget_usd=5,epochs=12,max_seconds=9000,max_updates=20000,batch_size=32,gradient_accumulation=1)
 settings_guard(c)
 for key,value in [('budget_usd',5.01),('epochs',13),('max_seconds',9001),('max_updates',20001),('architecture','convnext_tiny'),('batch_size',128)]:
  bad=dict(c);bad[key]=value
  with pytest.raises(ValueError):settings_guard(bad)

def test_continuation_requires_exact_checkpoint_and_unselected_campaign(tmp_path):
 from src.training.resume_c9 import verify_continuation
 from src.training.trainer_c9 import fingerprint
 from src.training.multidataset import digest
 from src.chess.labels import CLASS_NAMES
 c={'pretrained_sha256':'parent'};p=tmp_path/'last.pt';p.write_bytes(b'checkpoint')
 (tmp_path/'one_attempt_claim.json').write_text('{}')
 s=dict(updates=200,plan_hash=fingerprint(c),class_names=list(CLASS_NAMES),pretrained_sha256='parent',optimizer_state_dict={},rng={},scheduler_state_dict={},next_batch=200)
 a=dict(continuation_authorized=True,plan_hash=fingerprint(c),checkpoint_sha256=digest(p),updates=200)
 assert verify_continuation(tmp_path,a,c,s)==p
 for bad in [dict(a,continuation_authorized=False),dict(a,checkpoint_sha256='wrong'),dict(a,updates=201),dict(a,plan_hash='wrong')]:
  with pytest.raises(ValueError):verify_continuation(tmp_path,bad,c,s)
 (tmp_path/'selection_lock.json').write_text('{}')
 with pytest.raises(ValueError,match='selected or tested'):verify_continuation(tmp_path,a,c,s)

def test_forbidden_holdout_unreviewed_and_label_conflict():
 from src.training.dataset_c9 import validate_board
 b=board(1);b.update(representation='b2-context1-224-v1',geometry=dict(status='human_verified',kind='verified_prepared_context'),annotation=dict(status='human_verified'),inclusion=dict(status='included'))
 validate_board(b)
 for change in [dict(final_holdout=True),dict(geometry=dict(status='completed_unverified',kind='verified_prepared_context'))]:
  with pytest.raises(ValueError):validate_board(dict(b,**change))
 b['observations'][0]['class_id']=1
 with pytest.raises(ValueError):validate_board(b)

def test_internal_test_cannot_be_opened_for_training():
 from src.training.dataset_c9 import C9Squares
 from src.chess.labels import CLASS_NAMES
 m=dict(representation='b2-context1-224-v1',class_names=list(CLASS_NAMES),boards=[])
 with pytest.raises(ValueError,match='once-only'):C9Squares(m,'internal_test')

def test_comparison_alignment_rejects_shifted_squares_and_labels():
 from scripts.report_convnext_c9 import aligned
 a={'observations':[dict(board_id='frame',square='a8',truth=10)]}
 assert aligned(a,copy.deepcopy(a))
 for change in [dict(board_id='other'),dict(square='h8'),dict(truth=4)]:
  b=copy.deepcopy(a);b['observations'][0].update(change)
  with pytest.raises(ValueError,match='coordinates/labels'):aligned(a,b)

def test_final_evaluation_requires_exact_preexisting_selection_lock():
 from scripts.finalize_convnext_c9_cpu import verify_selection
 from src.training.trainer_c9 import fingerprint
 from src.chess.labels import CLASS_NAMES
 config={'seed':42};manifest={'boards':[]};state=dict(updates=1578,plan_hash=fingerprint(config),class_names=list(CLASS_NAMES))
 lock=dict(checkpoint_sha256='selected',plan_hash=fingerprint(config),dataset_fingerprint=fingerprint(manifest),selected_updates=1578)
 verify_selection(lock,config,manifest,state,'selected')
 with pytest.raises(ValueError):verify_selection(lock,config,manifest,state,'latest')
 with pytest.raises(ValueError):verify_selection(lock,{'seed':43},manifest,state,'selected')
 with pytest.raises(ValueError):verify_selection(lock,config,{'boards':['other']},state,'selected')
 with pytest.raises(ValueError):verify_selection(dict(lock,selected_updates=3682),config,manifest,state,'selected')

def test_corrections_and_exposure_metrics_preserve_physical_position():
 from src.training.evaluation_c9 import summarize_c9
 import numpy as np
 b=board(1);b.update(b2_training_placement_exposed=False,b2_exact_training_image_exposed=False,semantic_orientation='white_near_camera')
 rows=[(b,o) for o in b['observations']];z=np.full((64,13),-4.)
 for i,(_,o) in enumerate(rows):z[i,o['class_id']]=4.
 z[0,0]=-4.;z[0,4]=5.
 r=summarize_c9(rows,z.tolist())['metrics']
 assert r['exact_boards']==0 and r['corrections_per_board']['categories']['1']['count']==1
 assert r['empty_false_positives']==1
 assert r['b2_exposure_strata']['exposed_B2_training_placement'] is None
 assert len(r['worst_errors_by_position'])==1

def test_official_pretrain_cannot_silently_fallback(tmp_path):
 pytest.importorskip('timm')
 from src.training.convnext_c9 import make_model
 bad=tmp_path/'wrong.safetensors';bad.write_bytes(b'not pretrained weights')
 with pytest.raises(ValueError,match='Wrong pretrained'):make_model(bad)

def test_c9_native_rotation_adapter_matches_existing_review():
 from src.scanning.interim_video import decode_exact,pixel_hash
 from pathlib import Path
 import json
 inventory=Path('outputs/evaluation/convnext_c9/inventory_v1.json')
 if not inventory.exists():pytest.skip('Private media not installed')
 source=json.loads(inventory.read_text())['sessions'][0];idx=json.loads(Path(source['index_path']).read_text());row=dict(idx['original_frame_index'][0]);row['display_rotation_degrees']=row['encoded_rotation_degrees']
 image=decode_exact(source['source']['path'],row);packet=json.loads(Path('docs/evaluation/evidence_d2_geometry_review_v1/packet.json').read_text())
 assert pixel_hash(image)==packet['frames'][0]['frame']['decoded_pixel_sha256']

def test_interrupted_training_resumes_optimizer_and_scheduler(tmp_path,monkeypatch):
 """A tiny CPU toy validates the kernel state machine, not chess accuracy."""
 from src.training import trainer_c9 as kernel
 from src.training.convnext_c9 import MODEL_NAME
 from src.training.manifest_tools import labels_to_placement
 from src.training.multidataset import digest
 from torch import nn
 import hashlib
 torch.set_num_threads(2)
 boards=[]
 for i in range(100):
  labels=[0]*64
  for k in range(1,13):labels[(k+i%15)%64]=k
  b=board(i,fen=labels_to_placement(labels));b['source']['sha256']=hashlib.sha256(str(i).encode()).hexdigest();boards.append(b)
 class ToyDataset:
  def __init__(self,m,split,**kwargs):self.boards=m['boards'];self.rows=[(b,dict(square='a8',class_id=i%13)) for i,b in enumerate(self.boards)]
  def __len__(self):return len(self.rows)
  def set_epoch(self,e):pass
  def __getitem__(self,i):return torch.full((4,2,2),float(i%7)/7),i%13
 class Toy(nn.Module):
  def __init__(self):
   super().__init__();self.stem=nn.Conv2d(4,3,1);self.stages=nn.Sequential(nn.Conv2d(3,3,1));self.head=nn.Linear(3,13)
  def forward(self,x):return self.head(self.stages(self.stem(x)).mean((2,3)))
 def fake_evaluate(model,ds,**kwargs):
  model.eval()
  with torch.inference_mode():loss=float(torch.nn.functional.cross_entropy(model(ds[0][0][None]),torch.tensor([0])))
  return {'metrics':dict(macro_position_validation_loss=loss,overall_accuracy=0.,occupied_accuracy=0.,empty_false_positives=0,exact_boards=0)}
 monkeypatch.setattr(kernel,'C9Squares',ToyDataset);monkeypatch.setattr(kernel,'make_model',lambda path=None:Toy());monkeypatch.setattr(kernel,'evaluate',fake_evaluate)
 monkeypatch.setattr(kernel,'coverage_batches',lambda *args:[[0],[1],[2],[3]])
 parent=tmp_path/'fixture.pt';parent.write_bytes(b'unit fixture only');monkeypatch.setattr(kernel,'WEIGHT_SHA',digest(parent))
 cfg=dict(architecture=MODEL_NAME,representation='b2-context1-224-v1',seed=42,budget_usd=5,epochs=2,head_epochs=1,max_seconds=30,max_updates=20,batch_size=32,gradient_accumulation=1,
          workers=0,patience=4,checkpoint_every_updates=2,augmentation={},training_authorized=True,head_stem_lr=dict(stem=.001,head=.001),full_lr=dict(backbone=.0001,stem=.0001,head=.0001),weight_decay=.05)
 m={'boards':boards};kernel.run_training(m,cfg,parent,tmp_path/'full',device='cpu')
 def interrupt():raise InterruptedError('simulated after atomic checkpoint')
 with pytest.raises(InterruptedError):kernel.run_training(m,cfg,parent,tmp_path/'resumed',device='cpu',commit=interrupt)
 result=kernel.run_training(m,cfg,parent,tmp_path/'resumed',device='cpu',resume=tmp_path/'resumed/last.pt')
 a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
 assert result['checkpoint_reload_valid'] and result['updates']==8
 assert all(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]) for k in a['model_state_dict'])
 assert a['scheduler_state_dict']==b['scheduler_state_dict'] and a['source_exposure']==b['source_exposure']
 assert [v['step'].item() for v in a['optimizer_state_dict']['state'].values()]==[v['step'].item() for v in b['optimizer_state_dict']['state'].values()]

def test_archive_path_traversal_is_rejected(tmp_path):
 import tarfile,io
 from src.training.transfer_c9 import extract_archive
 from src.training.multidataset import digest
 archive=tmp_path/'bad.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  member=tarfile.TarInfo('../outside');member.size=1;t.addfile(member,io.BytesIO(b'x'))
 receipt=dict(archive_sha256=digest(archive),archive_bytes=archive.stat().st_size)
 with pytest.raises(ValueError,match='Unsafe'):extract_archive(archive,tmp_path/'data',receipt)
 assert not (tmp_path/'outside').exists()
