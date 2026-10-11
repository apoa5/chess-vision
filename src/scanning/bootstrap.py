"""Still-frame board bootstrap; no FEN, pieces, filenames or cross-frame geometry."""
import cv2
import numpy as np
from src.vision.board_detection import DetectionSettings,detect_board_corners
from src.vision.pipeline import rectify_with_config
from src.vision.perspective import order_corners
from src.vision.orientation import OrientationSettings,resolve_orientation,OrientationResult
from src.vision.grid_geometry import fit_grid
from src.scanning.replay import _geometry


def semantic_mapping(turns):
    """Canonical rank-8-first indices into the camera-view grid."""
    if type(turns) is not int or turns not in range(4):raise ValueError('Invalid orientation')
    return np.rot90(np.arange(64).reshape(8,8),turns)


def bootstrap_image(image,config,semantic_corners=None):
    assisted=semantic_corners is not None
    learned=None;learned_config=config.get('learned_markings',{});learned_enabled=learned_config.get('enabled',False)
    observations=None
    result={'schema_version':'guided_bootstrap_v1','status':'uncertain','review_required':True,'mode':'assisted' if assisted else 'automatic','geometry':None,'square_polygons':None,'recognition_connected':False}
    if assisted:
        semantic=np.asarray(semantic_corners,np.float32)
        # Validate winding, bounds and semantic identity before any sorting.
        h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),semantic)
        _geometry({'source':'manual','version':1,'corners':semantic.tolist(),'h_board_to_frame':h.ravel().tolist()},image.shape[1],image.shape[0])
        seed=order_corners(semantic)
        anchor=int(np.argmin(np.linalg.norm(seed-semantic[0],axis=1)))
        turns={3:0,0:1,1:2,2:3}[anchor]
    else:
        detection=detect_board_corners(image,DetectionSettings(**config.get('board_detection',{})))
        result['detection']={'success':detection.success,'reason':detection.reason,'score':detection.score,'corners':None if detection.corners is None else detection.corners.tolist()}
        seed=detection.corners;turns=None
        if learned_enabled:
            try:
                from src.scanning.marking_model import read_components
                from src.scanning.learned_markings import fit_marking_geometry,coarse_board_seed
                observations=read_components(image,threshold=learned_config.get('min_character_confidence',.80))
                if seed is None:seed=coarse_board_seed(image)
                if seed is not None:learned=fit_marking_geometry(observations,seed,image.shape)
                else:learned={'status':'uncertain','reason':'no_geometry_seed','observations':observations}
                result['learned_marking_evidence']=learned
            except (ImportError,OSError,ValueError,RuntimeError) as error:
                result['learned_marking_evidence']={'status':'unavailable','reason':str(error)}
        if seed is None:return result|{'reason':'localization_failed'}
    if learned and learned['status']=='candidate':
        semantic=np.asarray(learned['corners'],np.float32);seed=order_corners(semantic)
        anchor=int(np.argmin(np.linalg.norm(seed-semantic[0],axis=1)))
        turns={3:0,0:1,1:2,2:3}[anchor]
    rectified=rectify_with_config(image,seed,config)
    orientation=resolve_orientation(rectified,OrientationSettings(**config.get('orientation',{})))
    if learned and learned['status']=='candidate':
        if orientation.rotation_quarters_ccw is not None and orientation.rotation_quarters_ccw!=turns:
            return result|{'reason':'learned_and_template_orientation_conflict'}
        orientation=OrientationResult('resolved',turns,score=learned['inliers']/32,source='learned_border_markings')
    if not assisted and orientation.rotation_quarters_ccw is None:
        # A checkerboard repeats every two cells. Detector scores alone cannot
        # distinguish a full playing area from an equally alternating shifted seed.
        # Test neighboring one-cell offsets against independent border labels.
        x0,y0,x1,y1=rectified.board_bounds
        canvas_seed=np.float32([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]])
        inverse=np.linalg.inv(cv2.getPerspectiveTransform(seed,canvas_seed))
        step=(x1-x0)/8; hypotheses=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==dy==0:continue
                proposed=cv2.perspectiveTransform((canvas_seed+np.float32([dx*step,dy*step]))[:,None,:],inverse).reshape(4,2)
                if np.any(proposed<0) or np.any(proposed[:,0]>=image.shape[1]) or np.any(proposed[:,1]>=image.shape[0]):continue
                candidate=rectify_with_config(image,order_corners(proposed),config)
                evidence=resolve_orientation(candidate,OrientationSettings(**config.get('orientation',{})))
                if evidence.rotation_quarters_ccw is not None:hypotheses.append((evidence.score,proposed,candidate,evidence,dx,dy))
        hypotheses.sort(key=lambda h:h[0],reverse=True)
        result['neighboring_seed_resolved_count']=len(hypotheses)
        if hypotheses and (len(hypotheses)==1 or hypotheses[0][0]-hypotheses[1][0]>=.12):
            _,seed,rectified,orientation,dx,dy=hypotheses[0]
            result['seed_offset_squares']=[dx,dy]
        elif hypotheses:
            return result|{'reason':'competing_geometry_marking_hypotheses'}
    result['orientation_evidence']=orientation.metadata()
    if assisted:
        if orientation.rotation_quarters_ccw is not None and orientation.rotation_quarters_ccw!=turns:
            return result|{'reason':'manual_orientation_conflicts_with_markings'}
        orientation_source='manual_confirmed_by_markings' if orientation.rotation_quarters_ccw is not None else 'manual'
    else:
        turns=orientation.rotation_quarters_ccw
        if turns is None:return result|{'reason':'orientation_uncertain'}
        orientation_source='learned_border_markings' if learned and learned['status']=='candidate' else 'border_markings'
    grid=fit_grid(rectified.image,rectified.board_bounds,light_top_left=turns%2==0)
    result['grid_evidence_fraction']=grid['boundary_evidence_fraction']
    if grid['fit_status']!='fitted':return result|{'reason':'grid_rejected'}
    # Keep all source pixels; only map geometry back from the padded fitting canvas.
    x0,y0,x1,y1=rectified.board_bounds
    inverse=np.linalg.inv(cv2.getPerspectiveTransform(seed,np.float32([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]])))
    camera_polys=cv2.perspectiveTransform(np.asarray(grid['square_polygons'],np.float32).reshape(-1,1,2),inverse).reshape(64,4,2)
    ids=semantic_mapping(turns)
    polygons={f'{chr(97+c)}{8-r}':camera_polys[ids[r,c]].tolist() for r in range(8) for c in range(8)}
    lattice=cv2.perspectiveTransform(np.asarray(grid['intersections'],np.float32).reshape(-1,1,2),inverse).reshape(9,9,2)
    indices=[[(8,0),(8,8),(0,8),(0,0)],[(0,0),(8,0),(8,8),(0,8)],[(0,8),(0,0),(8,0),(8,8)],[(8,8),(0,8),(0,0),(8,0)]][turns]
    refined=np.asarray([lattice[r,c] for r,c in indices],np.float32)
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),refined)
    geometry={'h_board_to_frame':h.ravel().tolist(),'corners':refined.tolist(),'source':'manual' if assisted else 'markings','version':1}
    try:_geometry(geometry|{'source':'manual'},image.shape[1],image.shape[0])
    except ValueError:return result|{'reason':'refined_geometry_invalid_or_outside_frame'}
    if grid['boundary_evidence_fraction']<.25:return result|{'reason':'insufficient_grid_evidence'}
    # This is a conservative development guard, not a calibrated acceptance rule.
    step=np.mean(np.linalg.norm(np.roll(seed,-1,axis=0)-seed,axis=1))/8
    shift=float(np.max(np.linalg.norm(order_corners(refined)-seed,axis=1))/step)
    result['refinement_max_shift_squares']=shift
    if shift>.45:return result|{'reason':'grid_refinement_excessive_shift'}
    return result|{'status':'candidate','reason':'review_grid_before_use','geometry':geometry,'square_polygons':polygons,'orientation_source':orientation_source,'rotation_quarters_ccw':turns,'grid_method':grid['method']}


def draw_bootstrap(image,result):
    output=image.copy()
    for square,polygon in (result.get('square_polygons') or {}).items():
        points=np.rint(polygon).astype(np.int32);cv2.polylines(output,[points],True,(100,230,130),2)
        center=tuple(np.rint(points.mean(axis=0)).astype(int))
        cv2.putText(output,square,center,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4)
        cv2.putText(output,square,center,cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    return output
