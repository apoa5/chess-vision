"""Explicit offline FP32 runtime boundary; production never imports this module.

Input: normalized canonical a8..h1 RGB+mask float32, [64,4,224,224].
Output: float32 logits [64,13] in the unchanged Chess-Vision class order.
The runtime does not alter geometry, normalization, calibration or chess state.
The separate input-buffer helper reuses the original normalization kernel.
No implicit fallback is supported.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import SQUARES,digest

CHECKPOINT_SHA='45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb'
ARCHITECTURE='convnextv2_tiny.fcmae_ft_in22k_in1k'
PROFILE='b2-context1-224-v1'

class RuntimeUnavailable(RuntimeError):
    pass

def validate_inputs(array,squares):
    if tuple(squares)!=tuple(SQUARES):raise ValueError('Expected canonical a8..h1 square order; no implicit remapping')
    if not isinstance(array,np.ndarray) or array.dtype!=np.float32 or array.shape!=(64,4,224,224):raise ValueError('Expected float32 [64,4,224,224] normalized RGB+mask')
    if not np.isfinite(array).all():raise ValueError('Nonfinite input')
    mask=array[:,3]
    if not np.logical_or(mask==0,mask==1).all():raise ValueError('Target mask must remain binary')

def validate_config(config):
    if config.get('backend') not in ('pytorch_reference','pytorch_channels_last','onnx_cpu'):raise ValueError('Explicit supported backend required')
    if config.get('checkpoint_sha256')!=CHECKPOINT_SHA or config.get('representation')!=PROFILE or config.get('class_names')!=list(CLASS_NAMES):raise ValueError('Frozen model/profile/class contract required')
    if config.get('dtype')!='float32':raise ValueError('Only parity-tested FP32 is supported')
    if type(config.get('batch_size')) is not int or not 1<=config['batch_size']<=32:raise ValueError('CPU batch must be1..32')
    if type(config.get('threads')) is not int or not 1<=config['threads']<=4 or config.get('interop_threads')!=1:raise ValueError('Bounded CPU threads1..4 and interop1 required')
    if config['backend']=='onnx_cpu' and any(len(config.get(k,''))!=64 for k in ('onnx_sha256','onnx_metadata_sha256')):raise ValueError('Explicit pinned ONNX graph and metadata hashes required')

def semantic_weight_hash(model):
    """Hash logical tensor values independently of storage strides/layout."""
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        a=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def prepare_board_inputs(packed):
    """Avoid a64-array list+stack copy; use the original normalization kernel."""
    from src.vision.crop_experiments import tensor_from_packed
    if not isinstance(packed,np.ndarray) or packed.dtype!=np.uint8 or packed.shape!=(64,4,224,224):raise ValueError('Expected64 original packed uint8 crops')
    out=np.empty(packed.shape,dtype=np.float32)
    for i in range(64):out[i]=tensor_from_packed(packed[i])
    return out

class ConvNeXtRuntime:
    def __init__(self,config):
        validate_config(config);self.config=dict(config);self.last_timings={};start=time.perf_counter()
        if config['backend']=='onnx_cpu':self._load_onnx()
        else:self._load_pytorch()
        self.initialization_seconds=time.perf_counter()-start

    def _load_pytorch(self):
        try:import torch
        except ImportError as e:raise RuntimeUnavailable('PyTorch CPU backend unavailable; no fallback used') from e
        from src.training.convnext_c9 import make_model
        path=Path(self.config['checkpoint'])
        if digest(path)!=CHECKPOINT_SHA:raise ValueError('Checkpoint bytes differ; substitution prohibited')
        torch.set_num_threads(self.config['threads'])
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        state=torch.load(path,map_location='cpu',weights_only=True)
        if (state['architecture']!=ARCHITECTURE or state['input_channels']!=4 or state['input_size']!=224 or state['class_names']!=list(CLASS_NAMES)
            or state['representation']!=PROFILE or state['updates']!=1578):raise ValueError('Checkpoint metadata differs')
        self.model=make_model().float().eval();self.model.load_state_dict(state['model_state_dict'],strict=True);self.model.requires_grad_(False)
        del state
        self.weight_values_sha256=semantic_weight_hash(self.model)
        if self.config['backend']=='pytorch_channels_last':
            self.model.to(memory_format=torch.channels_last)
            if semantic_weight_hash(self.model)!=self.weight_values_sha256:raise ValueError('Layout conversion changed weight values')

    def _load_onnx(self):
        try:import onnxruntime as ort
        except ImportError as e:raise RuntimeUnavailable('ONNX Runtime is optional; install requirements-inference-c11.txt in the isolated environment. No fallback used.') from e
        ort.disable_telemetry_events()
        path=Path(self.config['onnx_path']);metadata_path=Path(self.config['onnx_metadata'])
        if digest(metadata_path)!=self.config['onnx_metadata_sha256']:raise ValueError('ONNX metadata changed')
        meta=json.loads(metadata_path.read_text())
        if (meta['checkpoint_sha256']!=CHECKPOINT_SHA or meta['class_names']!=list(CLASS_NAMES) or meta['representation']!=PROFILE
            or meta['architecture']!=ARCHITECTURE or meta['dtype']!='float32'
            or meta['input_shape']!=['batch',4,224,224] or meta['output_shape']!=['batch',13]):raise ValueError('ONNX model identity differs')
        if digest(path)!=meta['onnx_sha256'] or meta['onnx_sha256']!=self.config['onnx_sha256']:raise ValueError('ONNX artifact changed')
        if 'CPUExecutionProvider' not in ort.get_available_providers():raise RuntimeUnavailable('ONNX CPUExecutionProvider unavailable; no fallback used')
        options=ort.SessionOptions();options.intra_op_num_threads=self.config['threads'];options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if len(inputs)!=1 or inputs[0].name!='square_inputs' or inputs[0].shape[1:]!=[4,224,224] or inputs[0].type!='tensor(float)':raise ValueError('ONNX input contract differs')
        if len(outputs)!=1 or outputs[0].name!='logits' or outputs[0].shape[-1]!=13 or outputs[0].type!='tensor(float)':raise ValueError('ONNX output contract differs')

    def predict_logits(self,array,*,squares=SQUARES):
        start=time.perf_counter();validate_inputs(array,squares);validation=time.perf_counter()-start
        scores=[];layout=0.;forward=0.;batch_size=self.config['batch_size']
        if self.config['backend']=='onnx_cpu':
            for lo in range(0,64,batch_size):
                t=time.perf_counter();batch=np.ascontiguousarray(array[lo:lo+batch_size]);layout+=time.perf_counter()-t
                t=time.perf_counter();scores.append(self.session.run(['logits'],{'square_inputs':batch})[0]);forward+=time.perf_counter()-t
        else:
            import torch
            with torch.inference_mode():
                for lo in range(0,64,batch_size):
                    t=time.perf_counter();batch=torch.from_numpy(array[lo:lo+batch_size])
                    if self.config['backend']=='pytorch_channels_last':batch=batch.contiguous(memory_format=torch.channels_last)
                    layout+=time.perf_counter()-t;t=time.perf_counter();scores.append(self.model(batch).cpu().numpy());forward+=time.perf_counter()-t
        result=np.concatenate(scores,axis=0)
        if result.dtype!=np.float32 or result.shape!=(64,13) or not np.isfinite(result).all():raise RuntimeError('Runtime returned invalid logits')
        self.last_timings=dict(validation_seconds=validation,layout_seconds=layout,model_forward_seconds=forward,total_seconds=time.perf_counter()-start)
        return result

def load_runtime(config):
    return ConvNeXtRuntime(config)
