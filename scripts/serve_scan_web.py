#!/usr/bin/env python3
"""Serve the built prototype, optionally with a user-provided trusted TLS certificate."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl
import json
import threading
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOOTSTRAP_LOCK = threading.Lock()
INFERENCE_SLOTS = threading.BoundedSemaphore(2)
SESSION_LOCK = threading.Lock()
from src.scanning.recognition import SessionRegistry,recognize_keyframe
RECOGNITION_SESSIONS=SessionRegistry()


class DevelopmentHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/photo/scan/status?'):
            from urllib.parse import urlsplit,parse_qs
            from src.scanning.assisted_photo import PhotoError
            manager=getattr(self.server,'scan_service',None)
            try:
                if manager is None:raise PhotoError('profile_disabled','Guided multi-view scanning is unavailable in this server profile.',status=503)
                sid=parse_qs(urlsplit(self.path).query).get('session_id',[''])[0]
                self.send_json(manager.status(sid))
            except PhotoError as error:self.send_json({'status':'error','code':error.code,'message':str(error)},error.status)
            return
        if self.path=='/api/photo/status':
            service=getattr(self.server,'photo_service',None)
            self.send_json({'enabled':service is not None,'processing_location':'this Python server',
                            'model':service.info() if service else None,
                            'multiview_enabled':getattr(self.server,'scan_service',None) is not None})
            return
        super().do_GET()

    def do_POST(self):
        if self.path in ('/api/photo/scan/session','/api/photo/scan/candidate','/api/photo/scan/finish','/api/photo/scan/cancel'):
            self.scan_post()
            return
        if self.path in ('/api/photo/recognize','/api/photo/confirm'):
            self.photo_post()
            return
        started=time.perf_counter()
        if self.path not in ("/api/bootstrap","/api/scan/session","/api/scan/keyframe"):
            self.send_error(404); return
        origin = self.headers.get("Origin")
        if origin and origin.split("://",1)[-1] != self.headers.get("Host"):
            self.send_error(403); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400); return
        expected="application/json" if self.path=="/api/scan/session" else "image/jpeg"
        limit=4096 if self.path=="/api/scan/session" else 12*1024*1024
        if not 0 < length <= limit or self.headers.get("Content-Type") != expected:
            self.send_error(413, "Expected JPEG up to 12 MiB"); return
        # Receive uploads before taking the CPU lock: a slow/aborted camera
        # upload must not prevent another request from finding the board.
        try:
            connection=getattr(self,'connection',None)
            if connection is not None:connection.settimeout(15)
            raw=self.rfile.read(length)
            if len(raw)!=length:raise ValueError('Incomplete request body')
        except (OSError,ValueError) as error:
            self.send_error(400,str(error)); return
        if self.path=="/api/scan/session":
            try:
                if json.loads(raw)!={}:raise ValueError('Session creation expects an empty object')
                with SESSION_LOCK:result=RECOGNITION_SESSIONS.create()
                self.send_json(result)
            except (ValueError,TypeError,OSError,RuntimeError) as error:
                self.send_error(422,str(error))
            return
        # One running inference and at most one waiting request. Wait for a
        # prior saved frame to finish rather than making the browser poll 429s.
        if not INFERENCE_SLOTS.acquire(blocking=False):
            self.send_error(429,"Analysis queue is full; try again shortly"); return
        acquired=False;queued=time.perf_counter()
        try:
            acquired=BOOTSTRAP_LOCK.acquire(timeout=25)
            if not acquired:
                self.send_error(429,"Waiting for board analysis timed out"); return
            processing=time.perf_counter()
            import cv2
            import numpy as np
            from src.scanning.bootstrap import bootstrap_image
            from src.utils.config import load_config
            cv2.setNumThreads(1)
            if self.path=="/api/scan/keyframe":
                metadata=self.headers.get('X-Scan-Observation','')
                if len(metadata)>32768:raise ValueError('Metadata exceeds limit')
                meta=json.loads(metadata)
                if not isinstance(meta,dict):raise ValueError('Expected observation object')
                with SESSION_LOCK:RECOGNITION_SESSIONS.validate(meta)
                result=recognize_keyframe(raw,meta)
                with SESSION_LOCK:RECOGNITION_SESSIONS.accept(meta)
            else:
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[0] * image.shape[1] > 32000000:
                    raise ValueError("Invalid or oversized image")
                corners = json.loads(self.headers["X-Board-Corners"]) if self.headers.get("X-Board-Corners") else None
                cv2.setNumThreads(1)
                result = bootstrap_image(image, load_config()|{'scan_fast_acquisition':True}, corners)
            result['timing']={'queue_ms':round((processing-queued)*1000,1),'processing_ms':round((time.perf_counter()-processing)*1000,1),'total_ms':round((time.perf_counter()-started)*1000,1)}
            self.send_json(result)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, cv2.error) as exc:
            self.send_error(422, str(exc))
        finally:
            if acquired:BOOTSTRAP_LOCK.release()
            INFERENCE_SLOTS.release()

    def scan_post(self):
        from src.scanning.assisted_photo import PhotoError
        from src.scanning.assisted_multiview import MAX_CANDIDATE_BYTES
        from src.scanning.multiview_inference import ScanCancelled
        manager=getattr(self.server,'scan_service',None)
        if manager is None:
            self.send_json({'status':'error','code':'profile_disabled','message':'Multi-view requires the enabled optimized assisted-photo profile.'},503);return
        origin=self.headers.get('Origin')
        if origin and origin.split('://',1)[-1]!=self.headers.get('Host'):
            self.send_json({'status':'error','code':'origin','message':'Use the same application/API origin.'},403);return
        slot=False;acquired=False;handed_off=False
        try:
            candidate=self.path.endswith('/candidate')
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:raise PhotoError('upload_size','Invalid request length.',status=400)
            if not 0<length<=(MAX_CANDIDATE_BYTES if candidate else 4096):raise PhotoError('upload_size','Scan frame/metadata exceeds its bounded limit.',status=413)
            mime=self.headers.get('Content-Type','')
            if mime not in (('image/jpeg','image/png') if candidate else ('application/json',)):
                raise PhotoError('unsupported_image','Expected a JPEG/PNG candidate or JSON metadata.',status=415)
            if candidate:
                slot=INFERENCE_SLOTS.acquire(blocking=False)
                if not slot:raise PhotoError('queue_full','Analysis queue is busy. This candidate can be skipped.',status=429)
            self.connection.settimeout(15);raw=self.rfile.read(length)
            if len(raw)!=length:raise PhotoError('upload_incomplete','Interrupted scan upload.',status=400)
            if candidate:
                acquired=BOOTSTRAP_LOCK.acquire(timeout=25)
                if not acquired:raise PhotoError('queue_full','Geometry processing is busy.',status=429)
                sid=self.headers.get('X-Scan-Session','');frame_id=self.headers.get('X-Scan-Frame','')
                try:stamp=float(self.headers.get('X-Scan-Timestamp',''))
                except ValueError:raise PhotoError('candidate_metadata','Invalid frame timestamp.')
                result=manager.candidate(sid,frame_id,stamp,raw,mime)
            else:
                try:payload=json.loads(raw)
                except (TypeError,ValueError):raise PhotoError('invalid_metadata','Invalid scan JSON.',status=400)
                if not isinstance(payload,dict):raise PhotoError('invalid_metadata','Expected scan metadata object.')
                if self.path.endswith('/session'):
                    result=manager.create(payload)
                else:
                    sid=payload.get('session_id')
                    if not isinstance(sid,str) or len(sid)>80:raise PhotoError('invalid_metadata','Invalid scan session.')
                    if self.path.endswith('/cancel'):
                        result=manager.cancel(sid)
                    else:
                        slot=INFERENCE_SLOTS.acquire(blocking=False)
                        if not slot:raise PhotoError('queue_full','Recognition queue is full. Retry Finish Scan shortly.',status=429)
                        manager.claim_finish(sid,payload.get('view_budget',3))
                        job_queued=time.perf_counter()
                        def worker():
                            cpu=False
                            try:
                                cpu=BOOTSTRAP_LOCK.acquire(timeout=25)
                                if not cpu:raise PhotoError('queue_full','Recognition queue timed out. Retry the scan.')
                                manager._update(sid,queue_seconds=time.perf_counter()-job_queued)
                                manager.process(sid)
                            except ScanCancelled:manager._update(sid,phase='cancelled')
                            except PhotoError as error:manager._update(sid,phase='failed',error={'code':error.code,'message':str(error)})
                            except (ValueError,TypeError,OSError,RuntimeError,ImportError,MemoryError):
                                manager._update(sid,phase='failed',error={'code':'recognition_failed','message':'Multi-view recognition failed. No fallback checkpoint was used; retry or use single-photo mode.'})
                            finally:
                                with manager.lock:
                                    if sid in manager.sessions:manager.sessions[sid]['frames'].clear()
                                if cpu:BOOTSTRAP_LOCK.release()
                                INFERENCE_SLOTS.release()
                        threading.Thread(target=worker,name='chess-scan-'+sid[:8],daemon=True).start()
                        handed_off=True
                        self.send_json({'session_id':sid,'phase':'queued'},202);return
            self.send_json(result)
        except PhotoError as error:
            try:self.send_json({'status':'error','code':error.code,'message':str(error)},error.status)
            except (BrokenPipeError,ConnectionResetError):pass
        except (BrokenPipeError,ConnectionResetError):pass
        except (ValueError,TypeError,KeyError,OSError,RuntimeError):
            try:self.send_json({'status':'error','code':'scan_failed','message':'Scan request could not complete; retry or use single-photo mode.'},422)
            except (BrokenPipeError,ConnectionResetError):pass
        finally:
            if acquired:BOOTSTRAP_LOCK.release()
            if slot and not handed_off:INFERENCE_SLOTS.release()

    def photo_post(self):
        """Bound opt-in uploads before reading; share the existing CPU queue."""
        from src.scanning.assisted_photo import PhotoError,MAX_UPLOAD,confirm_position
        service=getattr(self.server,'photo_service',None)
        if service is None:
            self.send_json({'status':'error','code':'profile_disabled','message':'Assisted photo recognition is off. Start the server with --assisted-photo.'},503)
            return
        origin=self.headers.get('Origin')
        if origin and origin.split('://',1)[-1]!=self.headers.get('Host'):
            self.send_json({'status':'error','code':'origin','message':'Use the same server for the application and API.'},403)
            return
        inference=self.path.endswith('/recognize');slot=False;acquired=False
        started=time.perf_counter()
        try:
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:raise PhotoError('upload_size','Invalid request length.',status=400)
            limit=MAX_UPLOAD if inference else 16384
            if not 0<length<=limit:raise PhotoError('upload_size','Photographs must be up to 12 MiB; confirmation JSON must be up to 16 KiB.',status=413)
            content_type=self.headers.get('Content-Type','')
            if content_type not in (('image/jpeg','image/png') if inference else ('application/json',)):
                raise PhotoError('unsupported_image','Expected JPEG/PNG photograph or JSON confirmation.',status=415)
            if inference:
                slot=INFERENCE_SLOTS.acquire(blocking=False)
                if not slot:raise PhotoError('queue_full','Another recognition request is waiting. Please retry shortly.',status=429)
            self.connection.settimeout(15)
            raw=self.rfile.read(length)
            if len(raw)!=length:raise PhotoError('upload_incomplete','Upload interrupted. Please retry.',status=400)
            if inference:
                queued=time.perf_counter();acquired=BOOTSTRAP_LOCK.acquire(timeout=25)
                if not acquired:raise PhotoError('queue_full','Recognition is busy. Retry shortly.',status=429)
                try:manual=json.loads(self.headers.get('X-Photo-Geometry','{}'))
                except (TypeError,ValueError):raise PhotoError('invalid_geometry','Invalid manual alignment.')
                if len(self.headers.get('X-Photo-Geometry',''))>2048:raise PhotoError('invalid_geometry','Manual alignment is too large.')
                result=service.recognize(raw,content_type,manual)
                result['timings']['queue_seconds']=time.perf_counter()-queued-result['elapsed_seconds']
                result['request_seconds']=time.perf_counter()-started
            else:
                try:payload=json.loads(raw)
                except (TypeError,ValueError):raise PhotoError('invalid_position','Invalid confirmation JSON.',status=400)
                result=confirm_position(payload)
            self.send_json(result)
        except PhotoError as error:
            try:self.send_json({'status':'error','code':error.code,'message':str(error),**error.details},error.status)
            except (BrokenPipeError,ConnectionResetError):pass
        except (BrokenPipeError,ConnectionResetError):
            pass  # Client cancellation never turns a draft into a confirmation.
        except (ValueError,TypeError,KeyError,OSError,RuntimeError,ImportError) as error:
            print(f'Assisted photo request failed ({type(error).__name__}); no image retained.',file=sys.stderr)
            try:self.send_json({'status':'error','code':'recognition_failed','message':'ConvNeXt recognition could not complete. Check the server model/dependencies and retry; no fallback model was used.'},503)
            except (BrokenPipeError,ConnectionResetError):pass
        finally:
            if acquired:BOOTSTRAP_LOCK.release()
            if slot:INFERENCE_SLOTS.release()

    def send_json(self,result,status=200):
        payload=json.dumps(result).encode()
        self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def warm_scan_models():
    """Pay model imports/loading and first-forward costs before serving phone requests."""
    import torch
    from src.scanning.marking_model import load_reader
    from src.scanning.recognition import classifier
    reader=load_reader();pieces=classifier()
    with torch.inference_mode():
        reader(torch.zeros(128,1,48,36))
        pieces.model(torch.zeros(pieces.batch_size,4,pieces.input_size,pieces.input_size))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8443)
    parser.add_argument('--cert',type=Path)
    parser.add_argument('--key',type=Path)
    parser.add_argument('--assisted-photo',action='store_true',help='Enable the explicit ConvNeXt single-photo MVP profile; legacy scan default stays unchanged')
    parser.add_argument('--photo-runtime',choices=('optimized','reference'),default='optimized',help='Assisted photo backend; reference is for diagnostics')
    parser.add_argument('--scan-strategy',choices=('selected','sequential'),default='selected',help='Use measured multi-view policy or the unchanged C11 sequential reference')
    args=parser.parse_args()
    if bool(args.cert)!=bool(args.key):parser.error('--cert and --key are required together')
    directory=Path(__file__).resolve().parents[1]/'web'/'dist'
    if not (directory/'index.html').exists():parser.error('Build web/ first: npm --prefix web run build')
    photo_service=None
    if args.assisted_photo:
        from src.scanning.assisted_photo import PhotoService
        print('Loading assisted ConvNeXt photo profile (CC BY-NC 4.0 weights)…',flush=True)
        photo_service=PhotoService(reference=args.photo_runtime=='reference')
        photo_service.runtime()
    else:
        print('Preparing board and piece models…',flush=True)
        warm_scan_models()
    server=ThreadingHTTPServer((args.host,args.port),partial(DevelopmentHandler,directory=str(directory)))
    server.photo_service=photo_service
    server.scan_service=None
    if photo_service and args.photo_runtime=='optimized':
        from src.scanning.assisted_multiview import ScanManager
        server.scan_service=ScanManager(photo_service)
        if args.scan_strategy=='sequential':server.scan_service.config=dict(server.scan_service.config,strategy='sequential')
    scheme='http'
    if args.cert:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert,args.key)
        server.socket=context.wrap_socket(server.socket,server_side=True)
        scheme='https'
    print(f'Serving {scheme}://{args.host}:{args.port}; camera on a phone requires trusted HTTPS.',flush=True)
    if photo_service:print(f'Assisted photos: {scheme}://{args.host}:{args.port}/?mode=photo (processing on this computer)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
