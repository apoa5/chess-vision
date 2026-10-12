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
    def do_POST(self):
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

    def send_json(self,result):
        payload=json.dumps(result).encode()
        self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)

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
    args=parser.parse_args()
    if bool(args.cert)!=bool(args.key):parser.error('--cert and --key are required together')
    directory=Path(__file__).resolve().parents[1]/'web'/'dist'
    if not (directory/'index.html').exists():parser.error('Build web/ first: npm --prefix web run build')
    print('Preparing board and piece models…',flush=True)
    warm_scan_models()
    server=ThreadingHTTPServer((args.host,args.port),partial(DevelopmentHandler,directory=str(directory)))
    scheme='http'
    if args.cert:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert,args.key)
        server.socket=context.wrap_socket(server.socket,server_side=True)
        scheme='https'
    print(f'Serving {scheme}://{args.host}:{args.port}; camera on a phone requires trusted HTTPS.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
