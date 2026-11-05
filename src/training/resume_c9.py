"""Fail-closed authorization for continuing the interrupted C9 campaign."""
from pathlib import Path
from src.training.multidataset import digest


def verify_continuation(output, authorization, config, state):
    from src.training.trainer_c9 import fingerprint
    from src.chess.labels import CLASS_NAMES
    out = Path(output)
    if not authorization.get('continuation_authorized'):
        raise ValueError('Explicit continuation authorization required')
    if authorization['plan_hash'] != fingerprint(config):
        raise ValueError('Continuation configuration changed')
    if not (out / 'one_attempt_claim.json').exists():
        raise ValueError('Original campaign claim missing')
    if any((out / p).exists() for p in ('selection_lock.json', 'internal_test_one_attempt.json', 'c9_result.json')):
        raise ValueError('Campaign already selected or tested; continuation prohibited')
    if digest(out / 'last.pt') != authorization['checkpoint_sha256']:
        raise ValueError('Continuation checkpoint changed')
    if state['updates'] != authorization['updates'] or state['plan_hash'] != authorization['plan_hash']:
        raise ValueError('Continuation cursor or provenance changed')
    if state['class_names'] != list(CLASS_NAMES) or state['pretrained_sha256'] != config['pretrained_sha256']:
        raise ValueError('Continuation class or parent mismatch')
    for key in ('optimizer_state_dict', 'rng', 'scheduler_state_dict', 'next_batch'):
        if key not in state:
            raise ValueError('Incomplete resumable checkpoint: ' + key)
    return out / 'last.pt'
