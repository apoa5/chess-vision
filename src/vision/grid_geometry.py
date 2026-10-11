"""Image-supported square polygons on a padded board-plane canvas.

Hidden boundaries are inferred from the lattice; confidence is evidence strength,
not a claim that an occluded boundary was observed.
"""
import cv2
import numpy as np


def fit_grid(image, bounds, *, light_top_left=None):
    x0, y0, x1, y1 = bounds
    step = (x1-x0-1)/8
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Estimate the checker phase from square corners, away from piece centres.
    samples = np.array([[np.median(gray[int(y0+(r+.15)*step):int(y0+(r+.3)*step),
                                        int(x0+(c+.15)*step):int(x0+(c+.3)*step)])
                         for c in range(8)] for r in range(8)])
    parity = np.indices((8,8)).sum(axis=0)%2
    phase = 1 if np.median(samples[parity==1])-np.median(samples[parity==0]) >= 0 else -1

    if light_top_left is not None:
        phase = -1 if light_top_left else 1

    def lines(a, origin, perpendicular):
        positions = np.zeros((9,9), np.float32)
        strengths = np.zeros((9,8), np.float32)
        centres = perpendicular+(np.arange(8)+.5)*step
        edges = perpendicular+np.arange(9)*step
        # Recover the family offset/spacing first: a wrong seed can be half a cell off.
        full_profiles = []
        for band in range(8):
            start,end = int(perpendicular+(band+.08)*step),int(perpendicular+(band+.92)*step)
            profile = np.median(a[start:end],axis=0)
            full_profiles.append(profile[6:]-profile[:-6])
        full_profiles = np.asarray(full_profiles)
        scales,offsets = np.meshgrid(np.linspace(.85,1.15,61),np.arange(-.8*step,.8*step+1))
        scales,offsets = scales.ravel(),offsets.ravel()
        candidate = origin+offsets[:,None]+np.arange(1,8)[None,:]*step*scales[:,None]
        indices = np.clip(np.rint(candidate-3).astype(int),0,full_profiles.shape[1]-1)
        values = full_profiles[:,indices].transpose(1,0,2)
        signs = phase*np.where((np.arange(8)[:,None]+np.arange(1,8)[None,:])%2,1,-1)
        values = values*signs[None,:,:]
        scores = np.clip(values,-20,45).mean(axis=(1,2))-.012*np.abs(offsets)-2*np.abs(scales-1)
        best = int(np.argmax(scores))
        family_origin = origin+offsets[best]; family_step = step*scales[best]
        for boundary in range(9):
            expected = family_origin+boundary*family_step
            lo, hi = max(4,int(expected-.20*step)), min(a.shape[1]-4,int(expected+.20*step))
            evidence = []
            for band in range(8):
                start, end = int(perpendicular+(band+.08)*step), int(perpendicular+(band+.92)*step)
                profile = np.median(a[start:end,lo-3:hi+4], axis=0)
                derivative = profile[6:]-profile[:-6]
                sign = phase*(1 if (band+boundary)%2 else -1)
                evidence.append(derivative*sign if 0<boundary<8 else np.abs(derivative))
            evidence = np.asarray(evidence)
            # Joint line fit: piece edges in one band cannot bend an entire boundary.
            slopes, offsets = np.meshgrid(np.linspace(-.07,.07,29),np.arange(-.15*step,.15*step+1))
            slopes, offsets = slopes.ravel(), offsets.ravel()
            coords = expected+offsets[:,None]+slopes[:,None]*(centres-(perpendicular+4*step))
            indices = np.clip(np.rint(coords-lo).astype(int),0,hi-lo)
            values = evidence[np.arange(8)[None,:],indices]
            scores = np.clip(values,-20,45).mean(axis=1)-.05*np.abs(offsets)-2*np.abs(slopes)
            best = int(np.argmax(scores))
            intercept, slope = expected+offsets[best], slopes[best]
            positions[boundary] = intercept+slope*(edges-(perpendicular+4*step))
            indices = np.clip(np.rint(intercept+slope*(centres-(perpendicular+4*step))-lo).astype(int),0,hi-lo)
            strengths[boundary] = evidence[np.arange(8),indices]
            if np.count_nonzero(strengths[boundary]>=12)<3:
                positions[boundary] = expected
        return positions, strengths

    vertical, vx = lines(gray,x0,y0)
    horizontal, hy = lines(gray.T,y0,x0)
    points = np.zeros((9,9,2), np.float32)
    for r in range(9):
        for c in range(9):
            sx = (vertical[c,-1]-vertical[c,0])/(8*step)
            sy = (horizontal[r,-1]-horizontal[r,0])/(8*step)
            ax = vertical[c,0]-sx*y0
            ay = horizontal[r,0]-sy*x0
            x = (ax+sx*ay)/(1-sx*sy)
            points[r,c] = [x,ay+sy*x]
    yy,xx = np.mgrid[:9,:9]
    ideal = np.stack([xx,yy],axis=-1).astype(np.float32)
    mapping, inliers = cv2.findHomography(ideal[1:8,1:8].reshape(-1,2),points[1:8,1:8].reshape(-1,2),cv2.RANSAC,max(3,step*.06))
    if mapping is not None and int(inliers.sum())>=20:
        model = cv2.perspectiveTransform(ideal.reshape(-1,1,2),mapping).reshape(9,9,2)
        # Occluded outer edges follow the robust lattice; supported interior vertices
        # may depart slightly, without fitting piece contours as square edges.
        residual = points-model
        supported = np.linalg.norm(residual,axis=2)<step*.06
        points = model+residual*.35*supported[:,:,None]
        points[[0,8]] = model[[0,8]]; points[:,[0,8]] = model[:,[0,8]]
    valid_fit = not (np.any(np.diff(points[:,:,0],axis=1)<.4*step) or np.any(np.diff(points[:,:,1],axis=0)<.4*step))
    if not valid_fit:
        # Preserve a reviewable candidate rather than emitting folded polygons.
        yy,xx = np.meshgrid(y0+np.arange(9)*step,x0+np.arange(9)*step,indexing='ij')
        points = np.stack([xx,yy],axis=-1).astype(np.float32)
    polygons = [[points[r,c],points[r,c+1],points[r+1,c+1],points[r+1,c]]
                for r in range(8) for c in range(8)]
    return {'fit_status':'fitted' if valid_fit else 'rejected_seed_preview_only', 'version':'image_supported_grid_v1', 'intersections':points.tolist(),
            'square_polygons':np.asarray(polygons).tolist(),
            'boundary_evidence_fraction':float(np.mean(np.concatenate([vx.ravel(),hy.ravel()])>=12)),
            'boundary_evidence':{'vertical':vx.tolist(),'horizontal':hy.tolist()},
            'review_required':True,
            'method':'Local checker transitions; weak/occluded segments inferred from seed lattice. No lens calibration.'}


def polygon_crop(image, mask, polygon, padding):
    polygon = np.asarray(polygon,np.float32)
    left, top = np.floor(polygon.min(axis=0)).astype(int)
    right, bottom = np.ceil(polygon.max(axis=0)).astype(int)+1
    size = int(round((right-left+bottom-top)/2))+2*padding
    # Keep the training input size stable without resizing the source pixels.
    size = max(size, right-left, bottom-top)
    x = int(round((left+right-size)/2)); y = int(round((top+bottom-size)/2))
    crop = np.zeros((size,size,3),np.uint8); coverage = np.zeros((size,size),np.uint8)
    sx0,sy0 = max(0,x),max(0,y); sx1,sy1 = min(image.shape[1],x+size),min(image.shape[0],y+size)
    crop[sy0-y:sy1-y,sx0-x:sx1-x] = image[sy0:sy1,sx0:sx1]
    coverage[sy0-y:sy1-y,sx0-x:sx1-x] = mask[sy0:sy1,sx0:sx1]
    local = polygon-[x,y]
    target_mask = np.zeros((size,size),np.uint8)
    cv2.fillConvexPoly(target_mask,np.rint(local).astype(np.int32),255)
    valid = float((coverage[target_mask>0]>0).mean())
    tight = image[max(0,top):min(image.shape[0],bottom),max(0,left):min(image.shape[1],right)].copy()
    return crop, coverage, tight, [x,y,x+size,y+size], local.tolist(), valid
