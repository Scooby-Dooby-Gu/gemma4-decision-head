"""A small local server for trained Laya, so the payments demo can use it from another Python.

Laya needs PyTorch and lives in its own virtual environment, while the demo and the Gemma head
run on MLX. poc.payments.app starts this for you when Laya is set up; to run it by hand:

    .vendor/laya/.venv/bin/python -m poc.payments.laya_server --checkpoint poc/payments/laya/checkpoint

POST /answer {"q": "..."} -> {"answers": {question: {"choice", "confidence", "abstain"}}, "ms"}
GET /health -> {"ok": true}
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .bench import laya_engine

PORT = 8095


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="poc/payments/laya/checkpoint")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    _, ask = laya_engine(checkpoint=args.checkpoint)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/answer":
                return self._send(404, {"error": "not found"})
            try:
                q = str(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("q", ""))
                t = time.perf_counter()
                out = ask(q)
                self._send(200, {"answers": out["answers"], "ms": round((time.perf_counter() - t) * 1000, 1)})
            except Exception as e:  # a demo server: report the error to the caller
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"laya server on http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()  # one request at a time


if __name__ == "__main__":
    main()
