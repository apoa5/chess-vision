"""B.3 alignment, identity, deterministic training-only recipes and safety gates."""
import cv2
import numpy as np
import pytest

from src.vision.augmentation_experiments import (AugmentedCropDataset, FAMILIES, augment_packed,
                                               occupancy_metrics, sample_matrix, warp_packed)
from src.vision.crop_experiments import tensor_from_packed


def sample():
    packed = np.zeros((4, 224, 224), np.uint8)
    packed[3, 77:148, 77:148] = 1
    packed[:3] = packed[3]*255
    return packed


def test_forward_translation_moves_rgb_and_target_together():
    source = sample()
    matrix = np.array([[1, 0, 7], [0, 1, -4], [0, 0, 1]], float)
    actual, accepted = warp_packed(source, matrix)
    assert accepted
    np.testing.assert_array_equal(actual[0]//255, actual[3])
    assert actual[3, 73:144, 84:155].all()
    np.testing.assert_array_equal(source, sample())


@pytest.mark.parametrize('family', ['perspective_affine', 'framing'])
def test_sampled_geometry_preserves_alignment_orientation_and_target_guard(family):
    rng = np.random.default_rng(42)
    for _ in range(100):
        matrix = sample_matrix(family, rng)
        actual, accepted = warp_packed(sample(), matrix)
        assert accepted and np.linalg.det(matrix) > 0
        expected = cv2.warpPerspective(sample()[3], matrix, (224, 224), flags=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(actual[3], expected)
        assert set(np.unique(actual[3])) == {0, 1}
        disagreement = (actual[0] > 127) != actual[3].astype(bool)
        assert disagreement.sum() < 160  # interpolation differences only on boundary
        ys, xs = np.where(actual[3]); assert xs.min() > 55 and xs.max() < 170
        assert ys.min() > 55 and ys.max() < 170


def test_target_guard_rejects_clipping_and_does_not_relabel():
    actual, accepted = warp_packed(sample(), [[1, 0, 90], [0, 1, 0], [0, 0, 1]])
    assert not accepted
    np.testing.assert_array_equal(actual, sample())


@pytest.mark.parametrize('matrix', [np.diag([-1, 1, 1]), np.full((3, 3), np.nan), np.eye(2)])
def test_bad_transform_rejected(matrix):
    with pytest.raises(ValueError):
        warp_packed(sample(), matrix)


@pytest.mark.parametrize('family', FAMILIES)
def test_training_is_deterministic_finite_label_preserving_and_cache_immutable(family):
    packed = sample()[None]
    rows = [dict(crop_id='physical_square', split='train', class_id=10)]
    dataset = AugmentedCropDataset(rows, packed, split='train', seed=42, family=family)
    first, label = dataset[0]
    np.testing.assert_array_equal(first.numpy(), dataset[0][0].numpy())
    assert label == 10 and rows[0]['class_id'] == 10
    assert first.shape == (4, 224, 224) and np.isfinite(first.numpy()).all()
    assert set(np.unique(first[3].numpy())) <= {0, 1}
    np.testing.assert_array_equal(packed[0], sample())


@pytest.mark.parametrize('family', FAMILIES)
def test_validation_and_test_never_apply_augmentation(family):
    for split in ['val', 'test']:
        rows = [dict(crop_id='same', split=split, class_id=0)]
        dataset = AugmentedCropDataset(rows, sample()[None], split=split, family=family)
        dataset.set_epoch(4)
        np.testing.assert_array_equal(dataset[0][0].numpy(), tensor_from_packed(sample()))


@pytest.mark.parametrize('family', ['lighting', 'video_quality'])
def test_visual_augmentation_never_changes_mask_and_exercises_bounded_recipes(family):
    applied, kinds = 0, set()
    for seed in range(30):
        output, trace = augment_packed(sample(), family, np.random.default_rng(seed))
        np.testing.assert_array_equal(output[3], sample()[3])
        assert output.dtype == np.uint8
        applied += trace['applied']
        if 'quality_kind' in trace: kinds.add(trace['quality_kind'])
    assert 0 < applied < 30
    if family == 'video_quality': assert kinds == {0, 1, 2}


def test_empty_and_occupied_rates_use_correct_denominators():
    matrix = np.zeros((13, 13), int)
    matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[1, 2] = 8, 2, 1, 3, 1
    result = occupancy_metrics({'confusion_matrix': matrix})
    assert result['empty_false_positive_rate'] == .2
    assert result['missed_occupied_rate'] == .2
    assert result['occupied_identity_errors'] == 1


def test_validation_gate_rejects_false_pieces_even_with_higher_occupied_accuracy():
    from scripts.run_augmentation_experiments import safe_against
    matrix = np.eye(13, dtype=int)*10
    reference = dict(confusion_matrix=matrix.tolist(), empty_false_positives=0, missed_occupied=0, exact_boards=0,
                     by_orientation={'white_right': {'confusion_matrix': matrix.tolist()}})
    candidate = reference | {'empty_false_positives': 3, 'occupied_square_accuracy': 1.0}
    assert 'empty_false_positives' in safe_against(candidate, reference)


def test_selection_retains_anchor_when_both_seeds_do_not_show_gain():
    from scripts.run_augmentation_experiments import select_family
    matrix = np.eye(13, dtype=int)*10
    metrics = dict(confusion_matrix=matrix.tolist(), empty_false_positives=0, missed_occupied=0, exact_boards=0,
                   by_orientation={'white_right': {'confusion_matrix': matrix.tolist()}}, occupied_square_accuracy=1., macro_f1=1.)
    validation = {'b2_anchor': metrics}
    for seed in [42, 123]:
        for family in ['none', 'framing']:validation[f'{family}_seed{seed}'] = metrics
    result = select_family({'families': ['none', 'framing'], 'training': {'seeds': [42, 123]}}, validation)
    assert result['selected_family'] == 'retain_b2'


@pytest.mark.parametrize('index', [10, 11, 12])
def test_difficult_class_regressions_cannot_hide_behind_overall_accuracy(index):
    from scripts.run_augmentation_experiments import safe_against
    matrix = np.eye(13, dtype=int)*10
    reference = dict(confusion_matrix=matrix.tolist(), empty_false_positives=0, missed_occupied=0, exact_boards=0,
                     by_orientation={'white_right': {'confusion_matrix': matrix.tolist()}})
    regressed = matrix.copy(); regressed[index, index] -= 2; regressed[index, 7] += 2
    candidate = reference | {'confusion_matrix': regressed.tolist()}
    assert any('recall' in reason for reason in safe_against(candidate, reference))


def test_geometric_randomness_changes_across_epochs_without_changing_label():
    rows = [dict(crop_id='physical_square', split='train', class_id=10)]
    dataset = AugmentedCropDataset(rows, sample()[None], split='train', seed=42, family='framing')
    values = []
    for epoch in range(5):
        dataset.set_epoch(epoch)
        tensor, label = dataset[0]; assert label == 10
        values.append(tensor.numpy())
    assert any(not np.array_equal(values[0], value) for value in values[1:])
