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
    observations=None;raw_all=None;fast_seed=None;fast_attempt=None
    result={'implementation_version':'board_reader_multi_seed_v5','schema_version':'guided_bootstrap_v1','status':'uncertain','review_required':True,'mode':'assisted' if assisted else 'automatic','geometry':None,'square_polygons':None,'recognition_connected':False}
    if assisted:
        semantic=np.asarray(semantic_corners,np.float32)
        # Validate winding, bounds and semantic identity before any sorting.
        h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),semantic)
        _geometry({'source':'manual','version':1,'corners':semantic.tolist(),'h_board_to_frame':h.ravel().tolist()},image.shape[1],image.shape[0])
        seed=order_corners(semantic)
        anchor=int(np.argmin(np.linalg.norm(seed-semantic[0],axis=1)))
        turns={3:0,0:1,1:2,2:3}[anchor]
    else:
        # A marked wooden board can be registered directly from independently
        # recognized letters/numbers. Keep the expensive checker detector as a
        # fallback, never accept a color outline by itself.
        if learned_enabled and config.get('scan_fast_acquisition',False):
            from src.scanning.marking_model import read_components
            from src.scanning.learned_markings import fast_marking_geometry
            try:
                raw_all=read_components(image,threshold=learned_config.get('min_character_confidence',.80),retain_poses=True)
                fast_attempt=fast_marking_geometry(image,raw_all,learned_config.get('min_character_confidence',.80))
                if fast_attempt.get('reason')=='competing_alternative_seed_geometries':
                    return result|{'reason':fast_attempt['reason'],'learned_marking_evidence':fast_attempt}
                if fast_attempt['status']=='candidate':
                    learned=fast_attempt;fast_seed=order_corners(np.asarray(learned['corners'],np.float32))
            except (ImportError,OSError,ValueError,RuntimeError):
                pass  # The established detector/reader path still reports failures.
        if fast_seed is not None:
            seed=fast_seed;turns=None
            result['detection']={'success':True,'reason':None,'score':None,'corners':seed.tolist(),'source':'verified_marking_outlines'}
            result['marking_attempts']=[{'path':learned.get('acquisition_path','fast_marking_outlines'),'status':'candidate','reason':None}]
            result['learned_marking_evidence']=learned
        else:
            detection=detect_board_corners(image,DetectionSettings(**config.get('board_detection',{})))
            result['detection']={'success':detection.success,'reason':detection.reason,'score':detection.score,'corners':None if detection.corners is None else detection.corners.tolist()}
            seed=None if detection.corners is None else order_corners(detection.corners);turns=None
        if learned_enabled and fast_seed is None:
            try:
                from src.scanning.marking_model import read_components,read_plane_components
                from src.scanning.learned_markings import fit_marking_geometry,coarse_board_seed,coarse_board_seeds,fit_alternative_seeds
                threshold=learned_config.get('min_character_confidence',.80)
                if seed is None:seed=coarse_board_seed(image)
                if seed is None:
                    available=coarse_board_seeds(image)
                    seed=available[0] if available else None
                if seed is not None:
                    seed=order_corners(seed)
                    # Preserve the pre-update reader path before making any assumptions
                    # about how this board's characters face on individual edges.
                    if raw_all is None:raw_all=read_components(image,threshold=threshold,retain_poses=True)
                    best={}
                    for index,obs in enumerate(raw_all):
                        key=(obs.get('proposal_id',index),obs['label'])
                        if key not in best or best[key]['confidence']<obs['confidence']:best[key]=obs
                    observations=list(best.values())
                    learned=fit_marking_geometry(observations,seed,image.shape)
                    result['marking_attempts']=[{'path':'original_raw','status':learned['status'],'reason':learned.get('reason')}]
                    if learned['status']!='candidate':
                        alternatives=coarse_board_seeds(image)
                        learned=fit_alternative_seeds(raw_all,alternatives,image.shape)
                        result['marking_attempts'].append({'path':'alternative_outlines','status':learned['status'],'reason':learned.get('reason'),'seed_count':len(alternatives)})
                        if learned.get('reason')=='competing_alternative_seed_geometries':
                            return result|{'reason':learned['reason'],'learned_marking_evidence':learned}
                    if learned['status']!='candidate':
                        # Rectify the board alternatives too, not just a failed table outline.
                        plane_seeds=[]
                        for proposed in [*alternatives[:2],seed]:
                            if not any(np.max(np.linalg.norm(proposed-other,axis=1))<10 for other in plane_seeds):plane_seeds.append(proposed)
                        for proposed in plane_seeds:
                            plane=read_plane_components(image,proposed,threshold=threshold)
                            learned=fit_marking_geometry(plane,proposed,image.shape,max_seed_distance=7.)
                            observations=plane
                            result['marking_attempts'].append({'path':'rectified_plane','status':learned['status'],'reason':learned.get('reason')})
                            if learned['status']!='candidate':
                                edge=proposed[1]-proposed[0]
                                angle=float(np.degrees(np.arctan2(edge[1],edge[0])))
                                quarter=round(angle/90)
                                raw=raw_all
                                if abs(angle-90*quarter)<=22:
                                    raw=[o|{'glyph_seed_turns':(o['glyph_turns']-quarter)%4} if 'glyph_turns' in o else o for o in raw_all]
                                observations=plane+raw
                                learned=fit_marking_geometry(observations,proposed,image.shape,max_seed_distance=7.)
                                result['marking_attempts'].append({'path':'combined_pose','status':learned['status'],'reason':learned.get('reason')})
                            if learned['status']=='candidate':break
                else:learned={'status':'uncertain','reason':'no_geometry_seed','observations':[]}
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
                proposed=order_corners(proposed)
                candidate=rectify_with_config(image,proposed,config)
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
