"""B.2 preserves physical targets and the frozen cache/preprocessing contract."""
import numpy as np
import pytest

from src.vision.crop_experiments import CropSpec, packed_input, tensor_from_packed, transform_crop, CachedCropDataset
from src.vision.dataset_loader import preprocess_context_crop


def fixture():
    image = np.random.default_rng(4).integers(0, 256, (503, 503, 3), dtype=np.uint8)
    polygon = np.float32([[200.2, 201.5], [302.3, 200.8], [301.2, 302.7], [201.1, 301.9]])
    return image, polygon


@pytest.mark.parametrize('size', [224, 320])
@pytest.mark.parametrize('padding', [0, .5, 1, 2])
def test_cache_is_exact_shared_preprocessing_for_every_crop_and_resolution(size, padding):
    image, polygon = fixture()
    spec = CropSpec('test', size, padding)
    transformed, local = transform_crop(image, polygon, spec)
    actual = tensor_from_packed(packed_input(image, polygon, spec))
    np.testing.assert_array_equal(actual, preprocess_context_crop(transformed, local, size))
    assert actual.shape == (4, size, size)
    assert set(np.unique(actual[3])) <= {0, 1}


def test_original_pixels_and_target_are_unchanged_by_control():
    image, polygon = fixture()
    copy = image.copy()
    actual, target = transform_crop(image, polygon, CropSpec('control'))
    np.testing.assert_array_equal(actual, copy)
    np.testing.assert_array_equal(image, copy)
    np.testing.assert_array_equal(target, polygon)


def test_context_removal_and_detail_degradation_do_not_change_target_mask_or_scale():
    image, polygon = fixture()
    baseline = packed_input(image, polygon, CropSpec('baseline'))
    for spec in [CropSpec('suppressed', suppress_outer_context=True), CropSpec('detail', detail_factor=.25)]:
        transformed = packed_input(image, polygon, spec)
        np.testing.assert_array_equal(transformed[3], baseline[3])
        assert not np.array_equal(transformed[:3], baseline[:3])
    suppressed = packed_input(image, polygon, CropSpec('suppressed', suppress_outer_context=True))
    assert not suppressed[:3, :30].any()


@pytest.mark.parametrize('seed', [0, 42, 100])
def test_cached_photometric_augmentation_is_exact_frozen_augmentation(seed):
    image, polygon = fixture()
    packed = packed_input(image, polygon, CropSpec('control'))
    expected = preprocess_context_crop(image, polygon, rng=np.random.default_rng(seed))
    actual = tensor_from_packed(packed, rng=np.random.default_rng(seed))
    np.testing.assert_array_equal(actual, expected)


def test_larger_input_does_not_restore_removed_detail():
    image, polygon = fixture()
    small, _ = transform_crop(image, polygon, CropSpec('low224', detail_factor=.25))
    larger, _ = transform_crop(image, polygon, CropSpec('low320', input_size=320, detail_factor=.25))
    np.testing.assert_array_equal(small, larger)


def test_dataset_cannot_mix_split_rows_or_flagged_targets():
    cache = np.zeros((1, 4, 224, 224), np.uint8)
    row = dict(crop_id='one', class_id=0, split='test')
    with pytest.raises(ValueError, match='split'):
        CachedCropDataset([row], cache, split='train')
    with pytest.raises(ValueError, match='Unapproved'):
        CachedCropDataset([row | {'geometry_review_required': True}], cache, split='test')


def test_cached_training_is_epoch_deterministic_and_does_not_mutate_cache():
    image, polygon = fixture()
    packed = packed_input(image, polygon, CropSpec('cached'))[None]
    original = packed.copy()
    row = dict(crop_id='same_physical_target', class_id=10, split='train')
    ds = CachedCropDataset([row], packed, split='train', seed=42)
    first, label = ds[0]
    np.testing.assert_array_equal(first.numpy(), ds[0][0].numpy())
    ds.set_epoch(1)
    second, _ = ds[0]
    assert label == 10 and not np.array_equal(first.numpy()[:3], second.numpy()[:3])
    np.testing.assert_array_equal(first.numpy()[3], second.numpy()[3])
    np.testing.assert_array_equal(packed, original)


@pytest.mark.parametrize('shared', ['position', 'source_bytes', 'placement', 'session'])
def test_experiment_audit_rejects_known_leakage(shared):
    from scripts.run_crop_experiments import leakage_audit
    rows = [dict(crop_id='a', source_image='a', position_id='p1', split='train'),
            dict(crop_id='b', source_image='b', position_id='p2', split='test')]
    prepared = {'a': {'source_sha256': 'first'}, 'b': {'source_sha256': 'second'}}
    images = {'a': {}, 'b': {}}
    positions = {'p1': {'fen': 'position_one rest'}, 'p2': {'fen': 'position_two rest'}}
    if shared == 'position':
        rows[1]['position_id'] = 'p1'
    elif shared == 'source_bytes':
        prepared['b']['source_sha256'] = 'first'
    elif shared == 'placement':
        positions['p2']['fen'] = 'position_one different_metadata'
    else:
        images = {'a': {'session_id': 'same'}, 'b': {'session_id': 'same'}}
    with pytest.raises(ValueError, match='leakage'):
        leakage_audit(rows, prepared, images, positions)


def test_paired_uncertainty_uses_positions_and_occupied_targets():
    from scripts.analyze_crop_experiments import paired_positions
    rows = [dict(position_id='one', class_id=1), dict(position_id='one', class_id=0),
            dict(position_id='two', class_id=1), dict(position_id='three', class_id=1)]
    baseline = np.zeros((4, 13)); candidate = np.zeros((4, 13))
    baseline[:, 0] = 1; candidate[:, 0] = 1
    candidate[0] = 0; candidate[0, 1] = 1
    result = paired_positions(rows, baseline, candidate)
    assert result['physical_position_groups'] == 3
    assert result['by_position']['one']['occupied'] == 1
    assert result['by_position']['one']['occupied_accuracy_delta'] == 1
    assert result['leave_one_position_out_occupied_deltas']['one'] == 0
    assert result['leave_one_position_out_occupied_deltas']['two'] == .5


def test_each_adaptation_arm_copies_initial_weights_without_mutating_initializer():
    import torch
    from scripts.run_crop_experiments import load_model
    from src.vision.classifier import create_classifier
    initial = create_classifier(pretrained=False, mode='fine_tune').state_dict()
    key = next(iter(initial))
    original = initial[key].clone()
    arm = load_model({'model_state_dict': initial})
    assert next(arm.parameters()).data_ptr() != initial[key].data_ptr()
    with torch.no_grad():
        next(arm.parameters()).fill_(123)
    assert torch.equal(initial[key], original)
