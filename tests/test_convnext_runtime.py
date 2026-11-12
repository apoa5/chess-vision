"""Offline runtime contracts; no production imports or recognition tuning."""
import builtins,copy,hashlib,json
import numpy as np
import pytest
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import SQUARES
from src.vision.convnext_runtime import (CHECKPOINT_SHA,PROFILE,ConvNeXtRuntime,RuntimeUnavailable,
    validate_config,validate_inputs,prepare_board_inputs,semantic_weight_hash)
from scripts.benchmark_convnext_c11 import parity

def config(backend='pytorch_reference'):
    return dict(backend=backend,checkpoint='unused.pt',checkpoint_sha256=CHECKPOINT_SHA,
        representation=PROFILE,class_names=list(CLASS_NAMES),dtype='float32',batch_size=8,threads=2,interop_threads=1)

@pytest.mark.parametrize('change',[dict(backend='automatic'),dict(checkpoint_sha256='0'*64),dict(representation='other'),
    dict(class_names=list(CLASS_NAMES)[::-1]),dict(dtype='float16'),dict(batch_size=0),dict(batch_size=64),
    dict(threads=0),dict(threads=8),dict(interop_threads=2),dict(backend='onnx_cpu')])
def test_configuration_fails_closed(change):
    with pytest.raises(ValueError):validate_config(dict(config(),**change))

def test_input_contract_and_canonical_order():
    x=np.zeros((64,4,224,224),np.float32);validate_inputs(x,SQUARES)
    for bad in (x[:63],x[:,:3],x.astype(np.float64)):
        with pytest.raises(ValueError):validate_inputs(bad,SQUARES)
    with pytest.raises(ValueError):validate_inputs(x,SQUARES[::-1])
    x[0,3,0,0]=.5
    with pytest.raises(ValueError):validate_inputs(x,SQUARES)
    x[0,3,0,0]=0;x[0,0,0,0]=np.nan
    with pytest.raises(ValueError):validate_inputs(x,SQUARES)

def test_preallocated_normalization_is_bit_identical():
    from src.vision.crop_experiments import tensor_from_packed
    packed=np.random.default_rng(11).integers(0,256,(64,4,224,224),dtype=np.uint8);packed[:,3]=(packed[:,3]>127)
    before=hashlib.sha256(packed.tobytes()).hexdigest()
    actual=prepare_board_inputs(packed)
    for i in range(64):np.testing.assert_array_equal(actual[i],tensor_from_packed(packed[i]))
    assert hashlib.sha256(packed.tobytes()).hexdigest()==before
    validate_inputs(actual,SQUARES)
    with pytest.raises(ValueError):prepare_board_inputs(packed[:63])

def test_missing_optional_backend_never_falls_back(monkeypatch):
    original=builtins.__import__
    def blocked(name,*a,**k):
        if name=='onnxruntime':raise ImportError('deliberately unavailable')
        return original(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',blocked)
    cfg=dict(config('onnx_cpu'),onnx_sha256='a'*64,onnx_metadata_sha256='b'*64)
    with pytest.raises(RuntimeUnavailable,match='No fallback'):ConvNeXtRuntime(cfg)

def test_onnx_identity_is_checked_before_session_creation(tmp_path):
    pytest.importorskip('onnxruntime')
    path=tmp_path/'identity.json';path.write_text('{}')
    cfg=dict(config('onnx_cpu'),onnx_path=str(tmp_path/'nonexistent.onnx'),onnx_metadata=str(path),
        onnx_sha256='a'*64,onnx_metadata_sha256='b'*64)
    with pytest.raises(ValueError,match='metadata changed'):ConvNeXtRuntime(cfg)
    from src.vision.convnext_runtime import ARCHITECTURE
    meta=dict(checkpoint_sha256=CHECKPOINT_SHA,class_names=list(CLASS_NAMES)[::-1],representation=PROFILE,
        architecture=ARCHITECTURE,dtype='float32',input_shape=['batch',4,224,224],output_shape=['batch',13],onnx_sha256='a'*64)
    path.write_text(json.dumps(meta));cfg['onnx_metadata_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='identity differs'):ConvNeXtRuntime(cfg)

def test_wrong_checkpoint_bytes_rejected_before_deserialization(tmp_path):
    pytest.importorskip('torch');p=tmp_path/'wrong.pt';p.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='substitution prohibited'):ConvNeXtRuntime(dict(config(),checkpoint=str(p)))

def test_logical_weight_hash_is_stride_independent():
    torch=pytest.importorskip('torch');layer=torch.nn.Conv2d(4,3,3);h=semantic_weight_hash(layer)
    layer.to(memory_format=torch.channels_last)
    assert semantic_weight_hash(layer)==h
    with torch.no_grad():layer.weight[0,0,0,0]+=1
    assert semantic_weight_hash(layer)!=h

@pytest.mark.parametrize('batch',[1,7,8,16,32])
def test_runtime_batching_preserves_square_and_class_order(batch):
    # Stub executor tests transport/order only; real exported-model parity is a
    # separate required artifact, never substituted by this synthetic test.
    class Session:
        def run(self,names,feed):
            x=feed['square_inputs'];return [np.repeat(x[:,0,0,0,None],13,axis=1).astype(np.float32)]
    rt=object.__new__(ConvNeXtRuntime);rt.config=dict(config('onnx_cpu'),batch_size=batch);rt.session=Session()
    x=np.zeros((64,4,224,224),np.float32);x[:,0,0,0]=np.arange(64)
    z=rt.predict_logits(x)
    assert z.shape==(64,13)
    np.testing.assert_array_equal(z[:,0],np.arange(64))

def test_categorical_parity_is_independent_of_numerical_tolerance():
    ref=np.zeros((64,13),np.float32);ref[:,0]=1;ref[0,1]=1-1e-6
    candidate=ref.copy();candidate[0,1]=1+1e-6
    result=parity(ref,candidate,atol=.1,rtol=.1)
    assert result['numeric_within_tolerance'] and result['top1_disagreements']==1
    assert result['changed_squares']==['a8'] and result['board_changed']
    assert result['reference_placement']!=result['candidate_placement']
    assert not parity(ref,ref)['board_changed']

def test_invalid_output_logits_rejected():
    for bad in (np.zeros((64,12)),np.full((64,13),np.nan)):
        with pytest.raises(ValueError):parity(np.zeros((64,13)),bad)

@pytest.mark.parametrize('mode',['classes','dtype','nonfinite'])
def test_runtime_rejects_invalid_executor_output(mode):
    class Session:
        def run(self,names,feed):
            n=len(feed['square_inputs'])
            if mode=='classes':return [np.zeros((n,12),np.float32)]
            if mode=='dtype':return [np.zeros((n,13),np.float64)]
            return [np.full((n,13),np.nan,np.float32)]
    rt=object.__new__(ConvNeXtRuntime);rt.config=config('onnx_cpu');rt.session=Session()
    with pytest.raises(RuntimeError,match='invalid logits'):rt.predict_logits(np.zeros((64,4,224,224),np.float32))

def test_actual_export_is_pinned_and_has_four_channel_dynamic_contract():
    from pathlib import Path
    root=Path('outputs/evaluation/convnext_c11_v1');identity=root/'onnx_identity.json'
    if not identity.exists():pytest.skip('Real C11 export not present')
    onnx=pytest.importorskip('onnx');m=json.loads(identity.read_text());path=root/'convnext_fp32.onnx'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==m['onnx_sha256']
    model=onnx.load(path);onnx.checker.check_model(model)
    inp=model.graph.input[0];out=model.graph.output[0]
    assert inp.name=='square_inputs' and out.name=='logits'
    dims=inp.type.tensor_type.shape.dim
    assert dims[0].dim_param and [d.dim_value for d in dims[1:]]==[4,224,224]
    assert out.type.tensor_type.shape.dim[-1].dim_value==13

def test_recorded_logits_cannot_be_overwritten_on_replay(tmp_path,monkeypatch):
    import scripts.benchmark_convnext_c11 as runner
    import scripts.validate_convnext_c11 as validator
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(validator,'ROOT',tmp_path)
    for filename,fn in [('reference_c10_logits.npz',lambda:runner.reference('c10')),
        ('test_development_logits.npz',lambda:runner.full('test')),
        ('test_c10_logits.npz',lambda:validator.regression('test'))]:
        p=tmp_path/filename;p.write_bytes(b'preserve me')
        with pytest.raises(ValueError,match='already recorded'):fn()
        assert p.read_bytes()==b'preserve me'
