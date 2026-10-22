"""Offline representation matching for the development pilot; never ground truth."""
from fractions import Fraction
import cv2
import numpy as np


def display_image(image, rotation):
    """Apply PyAV/FFmpeg display rotation (negative 90 means clockwise)."""
    rotation = int(rotation)
    if rotation == 0:return image.copy()
    if rotation in (-90, 270):return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation in (90, -270):return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation in (180, -180):return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError('Only explicit right-angle display transforms supported')


def display_transform(width, height, rotation):
    if rotation == 0:return np.eye(3)
    if rotation in (-90, 270):return np.array([[0,-1,height-1],[1,0,0],[0,0,1]],float)
    if rotation in (90, -270):return np.array([[0,1,0],[-1,0,width-1],[0,0,1]],float)
    if rotation in (180, -180):return np.array([[-1,0,width-1],[0,-1,height-1],[0,0,1]],float)
    raise ValueError('Unsupported rotation')


def thumbnail(image):
    return cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(116,212),interpolation=cv2.INTER_AREA)


def correspondence_score(a,b):
    """Photometric-normalized whole-frame correlation, independent of detection."""
    a=a.astype(np.float32);b=b.astype(np.float32)
    a-=a.mean();b-=b.mean()
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    if denom < 1e-8:raise ValueError('Textureless image cannot establish correspondence')
    return float(np.sum(a*b)/denom)


def nearest_time(times,target):
    if not times:raise ValueError('Empty source timeline')
    return min(range(len(times)),key=lambda i:(abs(Fraction(times[i])-Fraction(target)),i))


def register_pair(high,low):
    """Independent cross-version feature registration; not board ground truth.

    Returns displayed high pixels -> low pixels. Deterministic OpenCV RNG.
    Reject weak registrations rather than compare grids using guessed scaling.
    """
    cv2.setRNGSeed(314159)
    scale=min(1.,848/max(high.shape[:2])); small=cv2.resize(high,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    orb=cv2.ORB_create(nfeatures=2500)
    a,da=orb.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None)
    b,db=orb.detectAndCompute(cv2.cvtColor(low,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:return {'status':'unavailable','reason':'missing_descriptors'}
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[x[0] for x in pairs if len(x)==2 and x[0].distance < .75*x[1].distance]
    if len(good)<12:return {'status':'unavailable','reason':'insufficient_matches','matches':len(good)}
    src=np.float32([a[x.queryIdx].pt for x in good]);dst=np.float32([b[x.trainIdx].pt for x in good])
    h,mask=cv2.findHomography(src,dst,cv2.RANSAC,2.)
    if h is None:return {'status':'unavailable','reason':'homography_failed'}
    keep=mask.ravel().astype(bool);n=int(keep.sum())
    pred=cv2.perspectiveTransform(src[:,None,:],h)[:,0,:];err=np.linalg.norm(pred-dst,axis=1)
    spread=np.ptp(dst[keep],axis=0) if n else np.zeros(2)
    if n<12 or n/len(good)<.5 or min(spread)<80:
        return {'status':'unavailable','reason':'weak_spatial_support','inliers':n,'matches':len(good)}
    # Actual rounded resize dimensions, not the pre-rounding nominal factor.
    h=h@np.diag([small.shape[1]/high.shape[1],small.shape[0]/high.shape[0],1.])
    return {'status':'registered','high_display_to_low_pixels':h.tolist(),'inliers':n,'matches':len(good),
            'median_inlier_error_low_px':float(np.median(err[keep])),
            'meaning':'Cross-version registration only; not reviewed board geometry.'}
