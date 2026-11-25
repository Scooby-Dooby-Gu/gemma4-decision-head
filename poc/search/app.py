"""Search POC, phase 1: a CRM search box that turns what you type into filters.

    python -m poc.search.app                 # http://127.0.0.1:8093
    python -m poc.search.app --no-llm        # without the Ollama comparison
    python -m poc.search.app --heads poc/search/heads   # with the trained heads

Two engines turn a query into filters, and the same SQLite query runs on either result:
  - head: the decision head on Gemma 4 E4B, one forward pass per filter question, no text
    generated. Listed option order and no blank-input prior, the best setup from phase 0.
  - llm:  Gemma 4 12B through Ollama, writing the filters out as JSON (the usual way).

POST /api/head {"q", "threshold", "exclude": {field: value}}
POST /api/llm  {"q", "exclude": {field: value}}
"exclude" maps a dismissed filter to the value it had, so it stays off until the value changes.
Both return {"chips", "filters", "search": {"count", "total", "rows", "where"}, "model_ms",
"db_ms"}; the head also returns "pending", what it leans towards but hasn't applied.
"""

import argparse
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from . import crm
from .filters import EAST, NONE, NOT_MENTIONED, QUESTIONS, WEST, to_filters

HERE = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434/api/chat"
LLM_MODEL = "gemma4:12b-mlx"
EMPTY = {"state": None, "kids": None, "age": None, "age_range": None, "marital": None}


def load_head(heads=None):
    from decision import DecisionModel

    dm = DecisionModel(debias_order=False)
    dm.answer("warm up", QUESTIONS)
    for s in dm.schemas.values():  # no blank-input prior: see README, phase 0 finding 1
        for view in s.views:
            view.prior = None
    if heads:
        dm.load_heads(heads)
    return dm


# --- the generative baseline --------------------------------------------------------------

STATE_NAMES = {**EAST, **WEST}
LLM_SYSTEM = (
    "You turn a search typed into a client database (CRM) into filters. The query may be "
    "unfinished, because it is sent as the person types, and may be in English, Spanish or "
    "French. Fill a field only when the query asks for it; otherwise use null.\n"
    "- state: the two-letter US state code (DC for Washington DC), from a state, city or nickname.\n"
    "- kids: has_kids or no_kids, about the clients themselves having children.\n"
    "- age_min / age_max: the clients' age range in whole years, inclusive. \"over 40\" is "
    "age_min 41; \"40+\" is age_min 40; \"in their 40s\" is 40 to 49; retirees or seniors are "
    "age_min 65; young adults are 18 to 34; middle-aged is 40 to 59.\n"
    "- marital: married, single, divorced or widowed.\n"
    "Reply with only a JSON object with exactly these keys: "
    '{"state": ..., "kids": ..., "age_min": ..., "age_max": ..., "marital": ...}'
)


def ask_llm(q):
    body = {"model": LLM_MODEL, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0},  # Ollama's MLX backend has no structured output
            "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": q}]}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.S)  # tolerate ```json fences or a stray sentence
    out = json.loads(m.group(0)) if m else {}
    lo, hi = out.get("age_min"), out.get("age_max")
    return {
        "state": out.get("state") if out.get("state") in STATE_NAMES else None,
        "kids": out.get("kids") if out.get("kids") in ("has_kids", "no_kids") else None,
        "age": "range" if lo is not None or hi is not None else None,
        "age_range": [lo, hi] if lo is not None or hi is not None else None,
        "marital": out.get("marital") if out.get("marital") in ("married", "single", "divorced", "widowed") else None,
    }, raw


# --- server ------------------------------------------------------------------------------

KIDS = {"has_kids": "Has kids", "no_kids": "No kids"}
AGE_KINDS = {"older_than": "Older than …", "younger_than": "Younger than …", "between": "Between … and …",
             "around": "Around …", "decade": "In their …s", "young": "Young adults",
             "middle_aged": "Middle-aged", "seniors": "Seniors"}


def age_label(rng):
    lo, hi = rng
    if hi is None:
        return f"Age {lo}+"
    if lo is None:
        return f"Age ≤ {hi}"
    return f"Age {lo}–{hi}"


def chips(filters, answers=None):
    """The applied filters as chips: {"field", "label", "value", "confidence"}."""
    conf = {}
    if answers:
        states = [answers[k] for k in ("state_east", "state_west") if answers[k]["choice"] == filters["state"]]
        conf = {"state": states[0]["confidence"] if states else None, "kids": answers["kids"]["confidence"],
                "age": answers["age"]["confidence"], "marital": answers["marital"]["confidence"]}
    out = []
    if filters["state"]:
        out.append({"field": "state", "label": STATE_NAMES[filters["state"]], "value": filters["state"]})
    if filters["kids"]:
        out.append({"field": "kids", "label": KIDS[filters["kids"]], "value": filters["kids"]})
    if filters["age_range"]:
        out.append({"field": "age", "label": age_label(filters["age_range"]), "value": filters["age_range"]})
    if filters["marital"]:
        out.append({"field": "marital", "label": filters["marital"].capitalize(), "value": filters["marital"]})
    for c in out:
        c["confidence"] = round(conf[c["field"]], 3) if conf.get(c["field"]) is not None else None
    return out


def pending(filters, answers):
    """What the head leans towards but hasn't applied: not confident enough, or no number yet."""
    out = []
    for key, a in answers.items():
        choice = a["choice"]
        if choice in (NONE, NOT_MENTIONED):
            continue
        field = "state" if key.startswith("state") else key
        if field == "age":
            applied = filters["age"] == choice
            label = AGE_KINDS[choice]
        else:
            applied = filters[field] == choice
            label = STATE_NAMES[choice] if field == "state" else KIDS.get(choice, choice.capitalize())
        if not applied:
            out.append({"field": field, "label": label, "confidence": round(a["confidence"], 3),
                        "abstain": a["abstain"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--heads", help="folder of trained heads (from poc.search.train_heads)")
    ap.add_argument("--no-llm", action="store_true", help="skip the Ollama comparison")
    args = ap.parse_args()

    db = crm.connect()
    print("loading the decision head ...", flush=True)
    dm = load_head(args.heads)
    head_lock, llm_lock, db_lock = Lock(), Lock(), Lock()

    def run_search(filters, exclude):
        """Drop filters the person dismissed, while they keep the value that was dismissed."""
        filters = dict(filters)
        for field, value in exclude.items():
            current = filters["age_range"] if field == "age" else filters.get(field)
            if current is not None and current == value:
                filters[field] = None
                if field == "age":
                    filters["age_range"] = None
        t = time.perf_counter()
        with db_lock:
            result = crm.search(db, filters)
        return filters, result, (time.perf_counter() - t) * 1000

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/health":
                # Trained heads are calibrated and less sure of themselves: 0.5 scored best on the
                # test queries, against 0.9 for the zero-label head (see README).
                return self._send(200, {"head": dm.backbone.name, "llm": None if args.no_llm else LLM_MODEL,
                                        "trained": bool(args.heads), "threshold": 0.5 if args.heads else 0.9})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                q = str(req.get("q", "")).strip()
                exclude = dict(req.get("exclude") or {})
                if self.path == "/api/head":
                    body = {"engine": "head", "q": q, "pending": []}
                    filters, model_ms, answers = dict(EMPTY), 0.0, None
                    if q:
                        t = time.perf_counter()
                        with head_lock:
                            answers = dm.answer(q, QUESTIONS)
                        model_ms = (time.perf_counter() - t) * 1000
                        filters = to_filters(q, answers, float(req.get("threshold", 0.9)))
                        body["pending"] = pending(filters, answers)
                elif self.path == "/api/llm":
                    if args.no_llm:
                        return self._send(404, {"error": "started with --no-llm"})
                    body = {"engine": "llm", "q": q}
                    filters, model_ms, answers = dict(EMPTY), 0.0, None
                    if q:
                        t = time.perf_counter()
                        with llm_lock:
                            filters, body["raw"] = ask_llm(q)
                        model_ms = (time.perf_counter() - t) * 1000
                else:
                    return self._send(404, {"error": "not found"})
                applied, result, db_ms = run_search(filters, exclude)
                body.update(chips=chips(applied, answers), filters=applied, search=result,
                            model_ms=round(model_ms, 1), db_ms=round(db_ms, 2))
                self._send(200, body)
            except Exception as e:  # a demo server: report the error to the page
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"search POC on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
