"""B.3 opt-in training-only perturbations of the frozen four-channel crop."""
import hashlib

import cv2
import numpy as np

from src.vision.crop_experiments import CachedCropDataset, tensor_from_packed

FAMILIES = ('none', 'legacy_visual', 'perspective_affine', 'framing', 'lighting', 'video_quality')

# Strengths are fixed for this preregistered screen; refuse metadata drift rather
# than silently ignoring an edited strength. Families can be enabled/disabled.
GEOMETRY_RECIPE = dict(apply_probability=.65, framing_rotation_degrees=5, framing_scale=[.95, 1.05],
                       framing_shift_fraction=.03, perspective_corner_jitter_fraction=.02,
                       affine_shear_degrees=2, target_guard_pixels=25, rgb_interpolation='linear',
                       mask_interpolation='nearest', rgb_border='replicate', mask_border='zero')
VISUAL_RECIPE = dict(lighting_gamma=[.85, 1.15], lighting_contrast=[.85, 1.15], lighting_exposure=[-.06, .06],
                     lighting_gradient=[-.15, .15], quality_gaussian_sigma=[.3, .75], quality_motion_kernel=3,
                     quality_jpeg_range=[75, 95], quality_noise_std_pixel_max=1.5)


def sample_matrix(family, rng, size=224):
    """Forward source→destination matrix: no flips, quarter turns or semantic remap."""
    if family not in ('perspective_affine', 'framing'):
        return np.eye(3, dtype=np.float64)
    center = (size-1)/2
    if family == 'framing':
        matrix = np.vstack([cv2.getRotationMatrix2D((center, center), rng.uniform(-5, 5), rng.uniform(.95, 1.05)), [0, 0, 1]])
        matrix[:2, 2] += rng.uniform(-.03, .03, 2)*size
    else:
        corners = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
        jitter = rng.uniform(-.02, .02, (4, 2))*size
        perspective = cv2.getPerspectiveTransform(corners, np.float32(corners+jitter))
        shear = np.eye(3)
        shear[0, 1], shear[1, 0] = np.tan(np.deg2rad(rng.uniform(-2, 2, 2)))
        shear[:2, 2] = -center*(shear[:2, :2].sum(axis=1)-1)
        matrix = perspective@shear
    return matrix


def warp_packed(packed, matrix):
    """Same homography, RGB linear/mask nearest, reject target/guard-band clipping.

    RGB uses replicated border (no invented reflected pieces). The small border
    wedges lie outside the central target region. The mask always has zero border.
    A 25-pixel guard around its bounding box is protected against clipping; this
    is a conservative heuristic, not a verified 3-D piece silhouette annotation.
    """
    if packed.ndim != 3 or packed.shape[0] != 4 or packed.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB+binary-mask cache')
    if not set(np.unique(packed[3])) <= {0, 1} or not packed[3].any():
        raise ValueError('Nonempty binary target mask required')
    height, width = packed.shape[1:]
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0:
        raise ValueError('Finite orientation-preserving homography required')
    ys, xs = np.where(packed[3])
    guard = np.float32([[xs.min()-25, ys.min()-25], [xs.max()+25, ys.min()-25],
                        [xs.max()+25, ys.max()+25], [xs.min()-25, ys.max()+25]])
    mapped = cv2.perspectiveTransform(guard[None], matrix)[0]
    if not np.isfinite(mapped).all() or (mapped < 0).any() or (mapped[:, 0] >= width).any() or (mapped[:, 1] >= height).any() or not cv2.isContourConvex(mapped):
        return packed.copy(), False
    rgb = cv2.warpPerspective(packed[:3].transpose(1, 2, 0), matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpPerspective(packed[3], matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if not mask.any():
        return packed.copy(), False
    return np.concatenate([rgb.transpose(2, 0, 1), mask[None]], axis=0), True


def augment_packed(packed, family, rng):
    if family not in FAMILIES:
        raise ValueError('Unknown augmentation family: '+family)
    output = packed.copy()
    trace = dict(family=family, applied=False, matrix=np.eye(3).tolist())
    if family in ('none', 'legacy_visual'):
        return output, trace
    if rng.random() >= .65:
        return output, trace
    if family in ('perspective_affine', 'framing'):
        matrix = sample_matrix(family, rng, packed.shape[1])
        output, accepted = warp_packed(packed, matrix)
        trace.update(applied=accepted, matrix=matrix.tolist(), rejected_guard=not accepted)
    elif family == 'lighting':
        rgb = output[:3].transpose(1, 2, 0).astype(np.float32)/255
        h, w = rgb.shape[:2]
        direction = rng.uniform(0, 2*np.pi)
        gradient = (np.linspace(-.5, .5, w)[None]*np.cos(direction)+np.linspace(-.5, .5, h)[:, None]*np.sin(direction))
        gamma, contrast, exposure, slope = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15), rng.uniform(-.06, .06), rng.uniform(-.15, .15)
        rgb = np.clip((np.power(rgb, gamma)-.5)*contrast+.5+exposure+slope*gradient[..., None], 0, 1)
        output[:3] = np.rint(rgb*255).astype(np.uint8).transpose(2, 0, 1)
        trace.update(applied=True, gamma=float(gamma), contrast=float(contrast), exposure=float(exposure), gradient=float(slope))
    else:
        rgb = output[:3].transpose(1, 2, 0)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            sigma = float(rng.uniform(.3, .75))
            rgb = cv2.GaussianBlur(rgb, (3, 3), sigma)
            trace['sigma'] = sigma
        elif kind == 1:
            kernel = np.zeros((3, 3), np.float32)
            if rng.random() < .5:
                kernel[1, :] = 1/3
            else:
                kernel[:, 1] = 1/3
            rgb = cv2.filter2D(rgb, -1, kernel)
        else:
            quality = int(rng.integers(75, 96))
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError('JPEG encoding failed')
            rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            trace['jpeg_quality'] = quality
        noise = rng.normal(0, rng.uniform(0, 1.5), rgb.shape)
        output[:3] = np.rint(np.clip(rgb.astype(np.float32)+noise, 0, 255)).astype(np.uint8).transpose(2, 0, 1)
        trace.update(applied=True, quality_kind=kind)
    return output, trace


class AugmentedCropDataset(CachedCropDataset):
    def __init__(self, *args, family='none', **kwargs):
        super().__init__(*args, **kwargs)
        if family not in FAMILIES:
            raise ValueError('Unknown augmentation family')
        self.family = family

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        packed = self.cache[index]
        if self.split != 'train' or self.family == 'none':
            array = tensor_from_packed(packed)
        elif self.family == 'legacy_visual':
            digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{row["crop_id"]}'.encode()).digest()
            array = tensor_from_packed(packed, rng=np.random.default_rng(int.from_bytes(digest[:8], 'big')), augmentation=self.augmentation)
        else:
            digest = hashlib.sha256(f'b3:{self.seed}:{self.epoch}:{row["crop_id"]}'.encode()).digest()
            packed, _ = augment_packed(packed, self.family, np.random.default_rng(int.from_bytes(digest[:8], 'big')))
            array = tensor_from_packed(packed)
        return torch.from_numpy(array), row['class_id']


def occupancy_metrics(metrics):
    matrix = np.asarray(metrics['confusion_matrix'])
    empty, occupied = int(matrix[0].sum()), int(matrix[1:].sum())
    false, missed = int(matrix[0, 1:].sum()), int(matrix[1:, 0].sum())
    return dict(empty_support=empty, occupied_support=occupied, empty_false_positives=false,
                empty_false_positive_rate=false/max(empty, 1), missed_occupied=missed,
                missed_occupied_rate=missed/max(occupied, 1),
                piece_detection_precision=float((occupied-missed)/max(occupied-missed+false, 1)),
                occupied_identity_errors=int(matrix[1:, 1:].sum()-np.trace(matrix[1:, 1:])))
