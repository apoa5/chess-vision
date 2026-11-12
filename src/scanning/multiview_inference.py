"""Bounded, shared-weight FP32 view/square transport; frozen C11 stays intact."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import numpy as np
from src.training.multidataset import SQUARES
from src.vision.convnext_runtime import validate_inputs

STRATEGIES = {
    'sequential': dict(batch_size=8, threads=4, workers=1),
    'combined8': dict(batch_size=8, threads=4, workers=1),
    'combined16': dict(batch_size=16, threads=4, workers=1),
    'combined32': dict(batch_size=32, threads=4, workers=1),
    'parallel2': dict(batch_size=8, threads=2, workers=2),
    'parallel3': dict(batch_size=8, threads=1, workers=3),
}


class ScanCancelled(RuntimeError):
    pass


class MultiViewRunner:
    """Only one caller; at most three workers sharing one immutable eval model.

    No worker changes PyTorch's process-wide thread budget or touches C11's
    mutable last_timings. Flat jobs retain explicit (view index, square offset).
    Chunking never constructs an unnecessary extra 192-input float buffer.
    """
    def __init__(self, runtime):
        if runtime.config['backend'] != 'pytorch_channels_last':
            raise ValueError('Multi-view experiment requires the C11 optimized FP32 model')
        self.runtime = runtime
        self.lock = threading.Lock()

    def predict(self, arrays, strategy='sequential', *, progress=None, cancelled=lambda:False):
        import torch
        if strategy not in STRATEGIES or not 1 <= len(arrays) <= 3:
            raise ValueError('Expected explicit bounded strategy and one to three views')
        for array in arrays: validate_inputs(array, SQUARES)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('The shared model is busy')
        cfg = STRATEGIES[strategy]
        original_threads = torch.get_num_threads()
        started = time.perf_counter()
        try:
            torch.set_num_threads(cfg['threads'])
            if strategy == 'sequential':
                output = []
                for i, array in enumerate(arrays):
                    if cancelled(): raise ScanCancelled('Scan cancelled')
                    output.append(self.runtime.predict_logits(array))
                    if progress: progress((i+1)*64, len(arrays)*64)
                return np.stack(output)
            jobs = [(view, lo) for view in range(len(arrays)) for lo in range(0, 64, cfg['batch_size'])]
            output = np.empty((len(arrays),64,13), np.float32)
            def forward(job):
                if cancelled(): raise ScanCancelled('Scan cancelled')
                view,lo = job
                with torch.inference_mode():
                    tensor = torch.from_numpy(arrays[view][lo:lo+cfg['batch_size']]).contiguous(memory_format=torch.channels_last)
                    values = self.runtime.model(tensor).cpu().numpy()
                if values.dtype != np.float32 or values.shape != (len(tensor),13) or not np.isfinite(values).all():
                    raise RuntimeError('Invalid multi-view logits')
                return view,lo,values
            completed = 0
            if cfg['workers'] == 1:
                for job in jobs:
                    view,lo,values = forward(job)
                    output[view,lo:lo+len(values)] = values
                    completed += len(values)
                    if progress: progress(completed,len(arrays)*64)
            else:
                # At most 24 small chunk jobs; at most three shared-model workers.
                with ThreadPoolExecutor(max_workers=cfg['workers'],thread_name_prefix='chess-view') as pool:
                    futures = [pool.submit(forward,job) for job in jobs]
                    try:
                        for future in as_completed(futures):
                            view,lo,values = future.result()
                            output[view,lo:lo+len(values)] = values
                            completed += len(values)
                            if progress: progress(completed,len(arrays)*64)
                    except BaseException:
                        for future in futures: future.cancel()
                        raise
            if cancelled(): raise ScanCancelled('Scan cancelled')
            return output
        finally:
            torch.set_num_threads(original_threads)
            self.last_seconds = time.perf_counter()-started
            self.lock.release()


def aggregate_views(logits, qualities, method='best_quality'):
    """D3 definitions: whole-frame quality choice, mean p, or softmax(mean z).

    Inputs must already be aligned to the same physical canonical squares.
    No probability multiplication, calibration, piece counts or chess repair.
    """
    from src.scanning.offline_fusion import softmax
    z = np.asarray(logits)
    if z.ndim != 3 or z.shape[1:] != (64,13) or not 1 <= len(z) <= 3 or not np.isfinite(z).all():
        raise ValueError('Expected one to three finite canonical [64,13] views')
    if len(qualities) != len(z) or not np.isfinite(qualities).all():
        raise ValueError('One finite non-prediction quality per view required')
    best = min(range(len(z)),key=lambda i:(-qualities[i],i))
    if method == 'best_quality': scores = np.array([softmax(v) for v in z[best]])
    elif method == 'mean_probabilities': scores = np.mean([[softmax(v) for v in view] for view in z],axis=0)
    elif method == 'mean_logits': scores = np.array([softmax(v) for v in z.astype(np.float64).mean(axis=0)])
    else: raise ValueError('Unregistered fusion method')
    predictions = z.argmax(axis=2)
    return dict(scores=scores, predictions=scores.argmax(axis=1), best_index=best,
                disagreement=(predictions != predictions[0]).any(axis=0), method=method,
                score_kind='uncalibrated_aggregation' if method!='best_quality' else 'uncalibrated_softmax')
