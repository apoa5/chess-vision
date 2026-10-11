#!/usr/bin/env python3
"""Serve the built prototype, optionally with a user-provided trusted TLS certificate."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl
import json
import threading
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOOTSTRAP_LOCK = threading.Lock()


class DevelopmentHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/bootstrap":
            self.send_error(404); return
        origin = self.headers.get("Origin")
        if origin and origin.split("://",1)[-1] != self.headers.get("Host"):
            self.send_error(403); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400); return
        if not 0 < length <= 12 * 1024 * 1024 or self.headers.get("Content-Type") != "image/jpeg":
            self.send_error(413, "Expected JPEG up to 12 MiB"); return
        if not BOOTSTRAP_LOCK.acquire(blocking=False):
            self.send_error(429, "Board analysis already running"); return
        try:
            import cv2
            import numpy as np
            from src.scanning.bootstrap import bootstrap_image
            from src.utils.config import load_config
            raw = self.rfile.read(length)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] * image.shape[1] > 32000000:
                raise ValueError("Invalid or oversized image")
            corners = json.loads(self.headers["X-Board-Corners"]) if self.headers.get("X-Board-Corners") else None
            cv2.setNumThreads(1)
            result = bootstrap_image(image, load_config(), corners)
            payload = json.dumps(result).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        except (ValueError, TypeError, cv2.error) as exc:
            self.send_error(422, str(exc))
        finally:
            BOOTSTRAP_LOCK.release()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


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
