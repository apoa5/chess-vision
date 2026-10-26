"""C.1 offline-only input prototypes. No classifier, geometry fitting or production hook."""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from src.vision.crop_experiments import CropSpec,packed_input,transform_crop
from src.vision.geometry_audit import context_transform,project
from src.vision.grid_geometry import polygon_crop
from src.vision.perspective import warp_board_with_context


@dataclass(frozen=True)
class RepresentationSettings:
    size: int = 224
    height_squares: float = 2.0
    raw_margin_squares: float = .35

    def __post_init__(self):
        if type(self.size) is not int or self.size<32 or not math.isfinite(self.height_squares) or not 0<self.height_squares<=4 or not math.isfinite(self.raw_margin_squares) or not 0<=self.raw_margin_squares<=2:
            raise ValueError('Invalid offline representation settings')


def outer_quad(geometry):
    p=np.asarray(geometry['source_square_polygons'],np.float32)
    if p.shape!=(64,4,2) or not np.isfinite(p).all():raise ValueError('Finite 64-square geometry required')
    q=np.float32([p[0,0],p[7,1],p[63,2],p[56,3]])
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q))<16:raise ValueError('Degenerate board quad')
    return q


def height_envelope(quad,target,width,height,height_squares):
    """Uncalibrated square-pixel/centered-principal-point pinhole proxy, not measured pose.

    Recover focal length from orthogonal board directions when well conditioned.
    Otherwise use an explicitly recorded image-up heuristic. Neither path is GT.
    """
    q=np.asarray(quad,np.float32);target=np.asarray(target,np.float32)
    H=cv2.getPerspectiveTransform(np.float32([[0,0],[8,0],[8,8],[0,8]]),q)
    a,b=H[:,0],H[:,1];cx,cy=width/2,height/2
    numerator=(a[0]-cx*a[2])*(b[0]-cx*b[2])+(a[1]-cy*a[2])*(b[1]-cy*b[2]);denominator=a[2]*b[2]
    reason='near_overhead_or_degenerate_vanishing_directions';focal=None
    if abs(denominator)>1e-12:
        f2=-numerator/denominator
        if f2>0 and (.2*math.hypot(width,height))**2<=f2<=(5*math.hypot(width,height))**2:
            focal=math.sqrt(f2);K=np.array([[focal,0,cx],[0,focal,cy],[0,0,1.]])
            B=np.linalg.inv(K)@H;scale=(np.linalg.norm(B[:,0])+np.linalg.norm(B[:,1]))/2
            r1,r2,t=B[:,0]/scale,B[:,1]/scale,B[:,2]/scale;n=np.cross(r1,r2);n/=np.linalg.norm(n)
            if n[2]>0:n=-n  # outward toward camera, not a metric gravity measurement
            xy=project(target,np.linalg.inv(H));base=xy[:,0,None]*r1+xy[:,1,None]*r2+t
            top=base+height_squares*n
            if np.all(top[:,2]>1e-6):
                uv=top@K.T;uv=uv[:,:2]/uv[:,2:]
                return np.vstack([target,uv]),dict(mode='assumed_pinhole_normal',focal_px=focal,height_squares=height_squares,
                    assumptions='square pixels; centered principal point; no lens distortion; visible plane; camera-facing normal',metric_pose_verified=False)
            reason='height_envelope_crosses_camera_plane'
        else:reason='unphysical_or_unstable_intrinsics_estimate'
    edges=np.linalg.norm(target-np.roll(target,-1,axis=0),axis=1);shift=np.array([0.,-height_squares*float(edges.mean())])
    return np.vstack([target,target+shift]),dict(mode='image_up_fallback',reason=reason,focal_px=None,height_squares=height_squares,
        assumptions='unknown normal direction; image-up only, may be wrong under camera roll',metric_pose_verified=False)


def raw_letterbox(image,target,envelope,settings):
    """Original non-warped pixels, aspect-preserving resize and a board-footprint mask."""
    edges=np.linalg.norm(target-np.roll(target,-1,axis=0),axis=1);margin=settings.raw_margin_squares*float(edges.mean())
    lo=np.floor(envelope.min(axis=0)-margin).astype(int);hi=np.ceil(envelope.max(axis=0)+margin).astype(int)+1
    w,h=hi-lo
    if w<=0 or h<=0 or w*h>100_000_000:raise ValueError('Invalid raw envelope extent')
    crop=np.zeros((h,w,3),np.uint8);valid=np.zeros((h,w),np.uint8)
    x0,y0=lo;x1,y1=hi;ix0,iy0=max(0,x0),max(0,y0);ix1,iy1=min(image.shape[1],x1),min(image.shape[0],y1)
    if ix1<=ix0 or iy1<=iy0:raise ValueError('Envelope outside source image')
    crop[iy0-y0:iy1-y0,ix0-x0:ix1-x0]=image[iy0:iy1,ix0:ix1];valid[iy0-y0:iy1-y0,ix0-x0:ix1-x0]=1
    s=settings.size;factor=min(s/w,s/h);nw,nh=max(1,round(w*factor)),max(1,round(h*factor));ox,oy=(s-nw)//2,(s-nh)//2
    rgb=np.zeros((s,s,3),np.uint8);mask=np.zeros((s,s),np.uint8);v=np.zeros((s,s),np.uint8)
    rgb[oy:oy+nh,ox:ox+nw]=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),(nw,nh),interpolation=cv2.INTER_AREA if factor<=1 else cv2.INTER_LINEAR)
    original_mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(original_mask,np.rint(target-lo).astype(np.int32),1)
    mask[oy:oy+nh,ox:ox+nw]=cv2.resize(original_mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
    v[oy:oy+nh,ox:ox+nw]=cv2.resize(valid,(nw,nh),interpolation=cv2.INTER_NEAREST)
    M=np.array([[nw/w,0,ox-x0*nw/w],[0,nh/h,oy-y0*nh/h],[0,0,1.]])
    return dict(packed=np.concatenate([rgb.transpose(2,0,1),mask[None]]),raw_to_input=M,
        source_support=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]),raw_bounds=[int(x0),int(y0),int(x1),int(y1)],
        valid_fraction=float(v.mean()),source_inside_fraction=float(valid.mean()),input_shape=[4,s,s],aspect_preserving=True)


def extract_representations(image,geometry,square,config,settings=None):
    settings=settings or RepresentationSettings()
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Source uint8 BGR required')
    ids=geometry['semantic_square_ids']
    if len(ids)!=64 or len(set(ids))!=64 or square not in ids:raise ValueError('Unique canonical square IDs required')
    target=np.float32(geometry['source_square_polygons'][ids.index(square)]);q=outer_quad(geometry)
    board=warp_board_with_context(image,np.float32(geometry['provenance']['source_corners']),config['board_size'],config['context_margin'])
    H=context_transform(geometry['provenance']['source_corners'],board.board_bounds);p=project(target,H)
    crop,coverage,_,bounds,local,valid=polygon_crop(board.image,board.valid_mask,p,config['initial_context_padding'])
    if valid<.99:raise ValueError('Source target clipped under frozen B.2 gate')
    result={}
    for name,spec in [('b2',CropSpec(**config['crop_spec'])),('expanded',CropSpec('expanded_context2_224',settings.size,2.))]:
        transformed,poly=transform_crop(crop,local,spec);packed=packed_input(crop,local,spec);shift=np.float32(local)[0]-poly[0]
        T=np.array([[1,0,-bounds[0]-shift[0]],[0,1,-bounds[1]-shift[1]],[0,0,1.]])
        M=np.diag([spec.input_size/transformed.shape[1],spec.input_size/transformed.shape[0],1.])@T@H
        support=project(np.float32([[0,0],[spec.input_size,0],[spec.input_size,spec.input_size],[0,spec.input_size]]),np.linalg.inv(M))
        result[name]=dict(packed=packed,raw_to_input=M,source_support=support,input_shape=list(packed.shape),aspect_preserving=False,
            context_shape=list(transformed.shape[:2]),target_valid_fraction=float(valid),mask_kind='board_plane_target_not_piece_silhouette')
    envelope,assumptions=height_envelope(q,target,image.shape[1],image.shape[0],settings.height_squares)
    result['taller']=raw_letterbox(image,target,envelope,settings);result['taller']['height_proxy']=assumptions
    # Branches intentionally share arrays: no redundant pixel storage/model implementation.
    result['two_branch']=dict(local=result['taller'],context=result['expanded'],classifier_implemented=False,
                              meaning='geometric height-envelope ROI plus wider context; not an isolated-piece detector')
    result['source_target_polygon']=target;result['source_board_quad']=q
    return result


def visible_bbox_coverage(bbox,source_support):
    """Fraction of assistant-marked visible BBOX, not segmented piece visibility or occlusion."""
    x0,y0,x1,y1=map(float,bbox)
    if x1<=x0 or y1<=y0:raise ValueError('Positive visible bbox required')
    box=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    support=cv2.convexHull(np.float32(source_support)).reshape(-1,2)
    area,_=cv2.intersectConvexConvex(box,support)
    return min(1.,max(0.,float(area)/((x1-x0)*(y1-y0))))


def jacobian_anisotropy(matrix,point):
    M=np.asarray(matrix,float);x,y=point;z=M[2]@[x,y,1.]
    if abs(z)<1e-9:raise ValueError('Projective horizon')
    n=M[:2]@[x,y,1.];J=(M[:2,:2]*z-n[:,None]*M[2,:2])/z**2;s=np.linalg.svd(J,compute_uv=False)
    return float(s[0]/s[-1]) if s[-1]>1e-12 else None
