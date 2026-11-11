"""Serve the decision model on Jev's /v1/systemone wire format (same as Von, Laya, Winnow).

    python serve.py                      # http://127.0.0.1:8092
    python serve.py --heads heads/       # with heads trained by train.py

    curl -s localhost:8092/v1/systemone -d '{"state": "...", "questions": {...}}'
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from decision import DecisionModel
from decision.backbone import DEFAULT_MODEL, GemmaBackbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--state-first", action="store_true",
                    help="state before the question; must match how the heads were trained")
    ap.add_argument("--heads", help="folder of trained heads")
    args = ap.parse_args()

    dm = DecisionModel(GemmaBackbone(args.model), state_first=args.state_first)
    if args.heads:
        dm.load_heads(args.heads)
    lock = Lock()  # one forward pass at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "loaded_model": args.model})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    answers = dm.answer(req["state"], req["questions"])
            except (KeyError, ValueError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, {"model": f"gemma-head ({args.model})", "answers": answers})

    print(f"decision model on http://{args.host}:{args.port}/v1/systemone")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
