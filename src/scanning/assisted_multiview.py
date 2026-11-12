"""Ephemeral guided-scan candidates, D1 selection and shared ConvNeXt evidence."""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,math,tempfile,threading,time,uuid
import cv2
import numpy as np
from src.chess.labels import CLASSES,CLASS_NAMES
from src.chess.board_state import BoardState
from src.chess.fen import placement_fen
from src.scanning.assisted_photo import PhotoError,decode_photo,ROOT
from src.scanning.multiview_inference import STRATEGIES,MultiViewRunner,ScanCancelled,aggregate_views
from src.scanning.video_keyframes import board_features,select_keyframes
from src.training.multidataset import SQUARES
from src.training.multidataset import digest
from src.vision.position_pipeline import prepare_raw_photo,PreparationError
from src.vision.geometry_audit import context_transform,project
from src.vision.dataset import chess_square_at
from src.vision.convnext_runtime import prepare_board_inputs
from scripts.run_square_evidence import make_inputs

MAX_CANDIDATE_BYTES=3*1024*1024
MAX_LONG_EDGE=2560
MAX_CANDIDATES=9
MAX_ATTEMPTS=24
MAX_SESSIONS=4
MAX_TOTAL_BYTES=64*1024*1024
TTL_SECONDS=600


def validate_geometry(geometry,width,height):
    p=np.asarray(geometry.get('source_square_polygons'),np.float32)
    ids=geometry.get('semantic_square_ids',[])
    if (not geometry.get('completed') or geometry.get('rotation_quarters_ccw') not in (0,1,2,3)
        or p.shape!=(64,4,2) or not np.isfinite(p).all() or len(ids)!=64 or set(ids)!=set(SQUARES)
        or p.min()<0 or p[...,0].max()>=width or p[...,1].max()>=height
        or any(not cv2.isContourConvex(q) or abs(cv2.contourArea(q))<1 for q in p)):
        raise PhotoError('unresolved_geometry','The complete grid and orientation could not be established. Try a clearer, more elevated view.')


def prepare_candidate(raw,mime,profile,settings):
    image,source=decode_photo(raw,mime)
    if max(image.shape[:2])>MAX_LONG_EDGE:
        raise PhotoError('candidate_size','Scan candidates must have a long edge no larger than 2560 pixels.')
    started=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='chessvision-scan-') as temporary:
            path=Path(temporary)/'frame.png'
            if not cv2.imwrite(str(path),image):raise OSError('Temporary frame encoding failed')
            rectified,grid,turns,provenance=prepare_raw_photo(path,profile['geometry'])
    except PreparationError as error:
        raise PhotoError(error.stage,{'localization':'Board not located; keep every playing corner visible.',
             'orientation':'Orientation unresolved; show the border markings or use single-photo manual alignment.',
             'grid':'Grid rejected; use a less extreme angle.'}.get(error.stage,'Unresolved geometry.')) from error
    except (ValueError,cv2.error) as error:
        raise PhotoError('unresolved_geometry','The board polygon or grid was invalid. Try another view.') from error
    transform=np.linalg.inv(context_transform(provenance['source_corners'],rectified.board_bounds))
    geometry=dict(completed=True,rotation_quarters_ccw=int(turns),provenance=provenance,
                  predicted_camera_orientation=['white_near_camera','white_left','black_near_camera','white_right'][turns],
                  source_square_polygons=project(grid['square_polygons'],transform).tolist(),
                  semantic_square_ids=[chess_square_at(r,c,turns) for r in range(8) for c in range(8)])
    validate_geometry(geometry,source['width'],source['height'])
    features=board_features(image,geometry,settings)
    return dict(source=source,geometry=geometry,features=features,geometry_seconds=time.perf_counter()-started,
                decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())


def square_records(z,context=None,scores=None):
    from src.scanning.offline_fusion import softmax
    probabilities=np.array([softmax(v) for v in z]) if scores is None else scores
    prediction=probabilities.argmax(1)
    return {s:dict(symbol=CLASSES[int(prediction[i])].symbol,class_name=CLASS_NAMES[int(prediction[i])],
                   confidence=float(probabilities[i].max()),probabilities=probabilities[i].tolist(),
                   logits=z[i].tolist() if z is not None else None,visibility='unknown',
                   context_valid_fraction=context[i] if context is not None else 1.) for i,s in enumerate(SQUARES)}


class ScanManager:
    def __init__(self,photo_service,config=None):
        self.photo=photo_service
        self.config=config or json.loads((ROOT/'configs/application/assisted_multiview_v1.json').read_text())
        if config is None:
            proof=ROOT/self.config['policy_evidence']
            if digest(proof)!=self.config['policy_evidence_sha256']:raise ValueError('Multi-view policy evidence changed')
            policy=json.loads(proof.read_text())
            if self.config['strategy']!=policy['selected_strategy'] or self.config['fusion_policy']!=policy['selected_fusion_policy']:
                raise ValueError('Multi-view runtime/fusion differs from the registered decision')
        if self.config['strategy'] not in STRATEGIES or self.config['fusion_policy'] not in ('best_quality','mean_logits','mean_probabilities'):
            raise ValueError('Explicit validated multiview policies required')
        self.settings=json.loads((ROOT/'configs/experiments/keyframes_d1.json').read_text())
        self.sessions={};self.lock=threading.RLock();self.runner=None

    def _cleanup(self):
        now=time.monotonic()
        for sid,s in list(self.sessions.items()):
            if now-s['created']>TTL_SECONDS:
                s['cancelled']=True;s['frames'].clear();s['timer'].cancel();self.sessions.pop(sid)

    def _expire(self,sid):
        with self.lock:
            s=self.sessions.pop(sid,None)
            if s:s['cancelled']=True;s['frames'].clear();s['result']=None

    def _get(self,sid):
        self._cleanup()
        if sid not in self.sessions:raise PhotoError('session_expired','Scan session expired. Start a new scan.',status=404)
        return self.sessions[sid]

    def create(self,payload):
        if not isinstance(payload,dict) or not {'unchanged_position'}<=set(payload)<= {'unchanged_position','source'} or payload.get('unchanged_position') is not True:
            raise PhotoError('position_attestation','Keep the physical board unchanged throughout one scan.')
        source=payload.get('source')
        if source is not None:
            required={'kind','filename','size_bytes','duration_ms','width','height'}
            if (not isinstance(source,dict) or set(source)!=required or source.get('kind')!='uploaded_video'
                or not isinstance(source.get('filename'),str) or not 1<=len(source['filename'])<=255
                or type(source.get('size_bytes')) is not int or not 0<source['size_bytes']<=1024**3
                or type(source.get('duration_ms')) not in (int,float) or not math.isfinite(source['duration_ms']) or not 0<source['duration_ms']<=1800000
                or any(type(source.get(k)) is not int or source[k]<=0 for k in ('width','height'))
                or source['width']*source['height']>32000000):
                raise PhotoError('video_metadata','Invalid or unsupported bounded video source metadata.')
            source=dict(source)
        with self.lock:
            self._cleanup()
            if len(self.sessions)>=MAX_SESSIONS:
                raise PhotoError('session_limit','Too many scan sessions. Cancel an old scan or wait for expiry.',status=429)
            sid=str(uuid.uuid4());self.sessions[sid]=dict(id=sid,created=time.monotonic(),phase='capturing',cancelled=False,
                source=source,frames=OrderedDict(),attempts=0,last_timestamp=-1.,seen=set(),diagnostics=[],selected_ids=[],result=None,
                inputs_prepared=0,completed_inputs=0,total_inputs=0,warnings=[])
            timer=threading.Timer(TTL_SECONDS,self._expire,args=(sid,));timer.daemon=True
            self.sessions[sid]['timer']=timer;timer.start()
            return dict(session_id=sid,max_candidates=MAX_CANDIDATES,max_attempts=MAX_ATTEMPTS,
                        max_candidate_bytes=MAX_CANDIDATE_BYTES,max_long_edge=MAX_LONG_EDGE,cadence_ms=3000,
                        classification_during_capture=False,expires_seconds=TTL_SECONDS)

    def candidate(self,sid,frame_id,timestamp_ms,raw,mime):
        if (not isinstance(frame_id,str) or not 1<=len(frame_id)<=120 or isinstance(timestamp_ms,bool)
            or not isinstance(timestamp_ms,(int,float)) or not math.isfinite(timestamp_ms) or timestamp_ms<0):
            raise PhotoError('candidate_metadata','Invalid candidate identity/timestamp.')
        if not 0<len(raw)<=MAX_CANDIDATE_BYTES:raise PhotoError('candidate_size','Scan frame exceeds 3 MiB.',status=413)
        with self.lock:
            s=self._get(sid)
            limit=s['source']['duration_ms'] if s['source'] else 120000
            if timestamp_ms>limit:raise PhotoError('candidate_metadata','Timestamp exceeds the source duration.')
            if s['phase']!='capturing':raise PhotoError('scan_state','This scan is no longer collecting frames.',status=409)
            if frame_id in s['seen'] or timestamp_ms<=s['last_timestamp']:raise PhotoError('stale_candidate','Duplicate or stale scan candidate.',status=409)
            if s['attempts']>=MAX_ATTEMPTS:raise PhotoError('candidate_limit','Candidate limit reached. Finish Scan or cancel.',status=409)
            s['attempts']+=1;s['last_timestamp']=timestamp_ms;s['seen'].add(frame_id)
        check_started=time.perf_counter()
        try:
            prepared=prepare_candidate(raw,mime,self.photo.profile,self.settings)
            reasons=prepared['features']['quality_rejection_reasons']
            if reasons:raise PhotoError('quality_rejected','Candidate rejected: '+', '.join(reasons))
            row=dict(prepared['features'],frame_id=frame_id,relative_timestamp_ms=timestamp_ms,
                     decoded_pixel_sha256=prepared['decoded_pixel_sha256'],geometry_state='completed_unverified',eligible=True)
            candidate=dict(id=frame_id,raw=raw,mime=mime,timestamp_ms=timestamp_ms,prepared=prepared,row=row,
                           binding=dict(session_id=sid,frame_id=frame_id,source=s['source'],source_sha256=hashlib.sha256(raw).hexdigest(),timestamp_ms=timestamp_ms))
        except PhotoError as error:
            with self.lock:
                s=self._get(sid);s['diagnostics'].append(dict(frame_id=frame_id,status='rejected',reason=error.code,message=str(error)))
                return dict(status='rejected',frame_id=frame_id,reason=error.code,message=str(error),retained_ids=list(s['frames']),attempts=s['attempts'],quality_check_seconds=time.perf_counter()-check_started)
        with self.lock:
            s=self._get(sid)
            if s['phase']!='capturing':raise PhotoError('scan_state','Scan stopped before this candidate completed.',status=409)
            pool=[*s['frames'].values(),candidate]
            selection=select_keyframes([c['row'] for c in pool],'quality_diverse',MAX_CANDIDATES,self.settings)
            keep=set(selection['selected']);new=OrderedDict((c['id'],c) for c in sorted(pool,key=lambda c:c['timestamp_ms']) if c['id'] in keep)
            total=sum(len(c['raw']) for other in self.sessions.values() if other is not s for c in other['frames'].values())+sum(len(c['raw']) for c in new.values())
            if total>MAX_TOTAL_BYTES:raise PhotoError('memory_limit','Scan cache is full. Finish or cancel another scan.',status=429)
            s['frames']=new;retained=frame_id in keep
            s['diagnostics'].append(dict(frame_id=frame_id,status='retained' if retained else 'redundant',reason=None if retained else 'duplicate_or_rank',geometry_status='completed_unverified',quality_score=row['quality_score'],orientation=prepared['geometry']['predicted_camera_orientation']))
            return dict(status='retained' if retained else 'redundant',frame_id=frame_id,retained_ids=list(new),attempts=s['attempts'],
                        geometry_status='completed_unverified',orientation=prepared['geometry']['predicted_camera_orientation'],quality_score=row['quality_score'],quality_check_seconds=time.perf_counter()-check_started)

    def cancel(self,sid):
        with self.lock:
            s=self._get(sid);s['cancelled']=True;s['phase']='cancelled';s['frames'].clear();s['result']=None
            s['timer'].cancel()
            # Remove immediately; active worker retains its own bounded references
            # and observes missing session as cancellation before the next chunk.
            self.sessions.pop(sid)
            return dict(status='cancelled')

    def status(self,sid):
        with self.lock:
            s=self._get(sid)
            return dict(session_id=sid,phase=s['phase'],attempts=s['attempts'],retained_ids=list(s['frames']),
                        candidate_count=len(s['frames']),selected_ids=s['selected_ids'],selected_count=len(s['selected_ids']),
                        inputs_prepared=s['inputs_prepared'],completed_inputs=s['completed_inputs'],total_inputs=s['total_inputs'],
                        warnings=list(s['warnings']),diagnostics=list(s['diagnostics']),result=s['result'],error=s.get('error'))

    def claim_finish(self,sid,view_budget=3):
        if type(view_budget) is not int or not 1<=view_budget<=3:
            raise PhotoError('view_budget','Choose a bounded maximum of one to three views.')
        with self.lock:
            s=self._get(sid)
            if s['phase']!='capturing':raise PhotoError('scan_state','Finish was already requested.',status=409)
            if not s['frames']:raise PhotoError('no_usable_frames','No usable frames. Keep all corners and border markings visible; move slowly or use single-photo manual alignment.')
            s['phase']='queued';s['view_budget']=view_budget

    def _update(self,sid,**fields):
        with self.lock:
            if sid in self.sessions and not self.sessions[sid]['cancelled']:self.sessions[sid].update(fields)

    def is_cancelled(self,sid):
        with self.lock:return sid not in self.sessions or self.sessions[sid]['cancelled']

    def process(self,sid):
        started=time.perf_counter();stage=time.perf_counter()
        self._update(sid,phase='selecting')
        with self.lock:
            s=self._get(sid)
            selection=select_keyframes([c['row'] for c in s['frames'].values()],'quality_diverse',s.get('view_budget',3),self.settings)
            selected=[s['frames'][i] for i in selection['selected']]
            submitted=s['attempts'];diagnostics=list(s['diagnostics']);s['selected_ids']=selection['selected'];s['total_inputs']=64*len(selected)
            queue_seconds=s.get('queue_seconds',0.)
        timings={'selection_seconds':time.perf_counter()-stage,'queue_seconds':queue_seconds};warnings=[]
        arrays=[];usable=[];contexts=[];stage=time.perf_counter()
        self._update(sid,phase='processing_inputs')
        for c in selected:
            if self.is_cancelled(sid):raise ScanCancelled('Scan cancelled')
            if hashlib.sha256(c['raw']).hexdigest()!=c['binding']['source_sha256'] or c['binding']['session_id']!=sid or c['binding']['frame_id']!=c['id']:
                raise PhotoError('geometry_binding','Candidate geometry is not bound to its exact source frame.')
            image,source=decode_photo(c['raw'],c['mime']);g=c['prepared']['geometry'];validate_geometry(g,source['width'],source['height'])
            if len(selected)==1:
                usable.append(c);contexts.append(None)
                continue  # Let the unchanged photo path extract this one view once.
            try:
                inputs=make_inputs(image,g,self.photo.profile['crop']);lookup={i['canonical_square']:i for i in inputs}
                packed=np.stack([lookup[q]['packed'] for q in SQUARES]);arrays.append(prepare_board_inputs(packed));contexts.append([lookup[q]['context_valid_fraction'] for q in SQUARES]);usable.append(c)
            except (ValueError,cv2.error):
                warnings.append('Selected frame '+c['id']+' has invalid/clipped inputs; omitted without replacement.')
            self._update(sid,inputs_prepared=len(arrays))
        timings['input_preparation_seconds']=time.perf_counter()-stage
        if not usable:raise PhotoError('no_usable_frames','All selected frame inputs were invalid. Please rescan.')
        self._update(sid,phase='recognizing',total_inputs=64*len(usable),warnings=warnings)
        stage=time.perf_counter()
        if len(usable)==1:
            # Actual unchanged MVP-1 path, not a substitute checkpoint.
            arrays.clear()
            result=self.photo.recognize(usable[0]['raw'],usable[0]['mime'])
            photo_timings=dict(result['timings']);contexts[0]=[result['squares'][q]['context_valid_fraction'] for q in SQUARES]
            z=np.asarray([[result['squares'][q]['logits'] for q in SQUARES]],np.float32)
            inference_strategy='single_photo_fallback';warnings.append('One view selected/usable: using the existing single-photo pathway.')
            self._update(sid,completed_inputs=64)
        else:
            if self.runner is None:self.runner=MultiViewRunner(self.photo.runtime())
            z=self.runner.predict(arrays,self.config['strategy'],cancelled=lambda:self.is_cancelled(sid),
                                  progress=lambda done,total:self._update(sid,completed_inputs=done,total_inputs=total))
            inference_strategy=self.config['strategy'];result=None
        timings['inference_seconds']=time.perf_counter()-stage
        if len(usable)==1:
            timings['single_photo_pipeline_seconds']=timings['inference_seconds']
            timings['inference_seconds']=photo_timings['total_seconds']
            timings['geometry_post_scan_seconds']=photo_timings['geometry_seconds']
            timings['single_photo_timings']=photo_timings
        else:timings['geometry_post_scan_seconds']=0.  # Geometry was bound during capture.
        if self.is_cancelled(sid):raise ScanCancelled('Scan cancelled')
        self._update(sid,phase='combining');stage=time.perf_counter()
        quality=[c['row']['quality_score'] for c in usable]
        policy=self.config['fusion_policy'] if len(usable)>1 else 'best_quality'
        aggregate=aggregate_views(z,quality,policy);best=aggregate['best_index'];reference=usable[best];g=reference['prepared']['geometry']
        per_view=[dict(frame_id=c['id'],timestamp_ms=c['timestamp_ms'],binding=c['binding'],geometry_status='completed_unverified',
                       orientation=c['prepared']['geometry']['predicted_camera_orientation'],quality=c['row']['quality_score'],
                       quality_metadata={k:c['row'].get(k) for k in ('board_sharpness','board_area_fraction','edge_ratio_proxy','view_descriptor','visibility','elevation_degrees')},
                       geometry_seconds_at_capture=c['prepared']['geometry_seconds'],
                       geometry=dict(source_corners=c['prepared']['geometry']['provenance']['source_corners'],
                            rotation_quarters_ccw=c['prepared']['geometry']['rotation_quarters_ccw'],canonical_square_order=list(SQUARES)),
                       squares=square_records(values,context)) for c,values,context in zip(usable,z,contexts)]
        if result is None:
            final_z=z[best] if policy=='best_quality' else z.mean(0) if policy=='mean_logits' else None
            squares=square_records(final_z,contexts[best],aggregate['scores'])
            symbols=[squares[q]['symbol'] for q in SQUARES]
            result=dict(schema_version='assisted_photo_result_v1',recognition_id=str(uuid.uuid4()),status='draft',confirmed=False,
                        image=reference['prepared']['source'],source_sha256=reference['binding']['source_sha256'],squares=squares,
                        square_order=list(SQUARES),provisional_placement=placement_fen(BoardState([symbols[i:i+8] for i in range(0,64,8)],'white_at_bottom')),
                        model=self.photo.info(),geometry=dict(status='completed_unverified',owner_verified=False,policy='legacy/default',
                            source_corners=g['provenance']['source_corners'],rotation_quarters_ccw=g['rotation_quarters_ccw'],
                            orientation_source=g['provenance']['orientation_source'],camera_orientation=g['predicted_camera_orientation'],
                            square_polygons=dict(zip(g['semantic_square_ids'],g['source_square_polygons']))))
        if len(usable)<3:warnings.append('Fewer than three usable distinct views were available; no rejected view was silently substituted.')
        if policy=='best_quality':warnings.append('Draft uses the whole best-quality view; fusion was not promoted by the registered development rule.')
        warnings.append('Geometry completion is not correctness. Review all squares; video-frame scores are correlated and uncalibrated.')
        timings['aggregation_seconds']=time.perf_counter()-stage;timings['post_scan_seconds']=time.perf_counter()-started
        result['aggregation']=dict(method=policy,score_kind=aggregate['score_kind'],raw_logits_kind='selected_view' if policy=='best_quality' else 'mean_logits' if policy=='mean_logits' else 'not_available_for_probability_mean')
        result['multiview']=dict(session_id=sid,source=s['source'],submitted_view_count=submitted,selected_view_count=len(selected),accepted_view_count=len(usable),
                    reference_frame_id=reference['id'],selected_strategy=inference_strategy,fusion_policy=policy,per_view=per_view,
                    selection=selection,diagnostics=diagnostics,warnings=warnings,disagreement_squares=[SQUARES[i] for i in np.flatnonzero(aggregate['disagreement'])],
                    geometry_unverified=True,timings=timings,geometry_at_capture_seconds=sum(c['prepared']['geometry_seconds'] for c in selected))
        result['elapsed_seconds']=timings['post_scan_seconds'];result['timings']=timings
        with self.lock:
            if sid in self.sessions and not self.sessions[sid]['cancelled']:
                s=self.sessions[sid];s['phase']='review_ready';s['result']=result;s['warnings']=warnings;s['frames'].clear()
        return result
