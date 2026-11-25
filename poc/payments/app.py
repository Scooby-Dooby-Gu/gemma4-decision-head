"""Card transaction search demo: type a search, filters update after every word.

    python -m poc.payments.app            # http://127.0.0.1:8094

Three engines turn the query into the same filters, and the same SQLite query runs on each:
  - head: the decision head on Gemma 4 E4B, trained on the generated examples (poc/payments/heads)
  - laya: Laya's multilingual encoder, trained on the same examples (poc/payments/laya/checkpoint),
          served by laya_server.py from Laya's own environment, which this app starts
  - jev:  TypeSafe's Jev through OpenRouter, untrained (a key in .env.local)
An engine whose files or key are missing is left out.

POST /api/search {"q", "engine", "threshold", "exclude": {field: value}}
  -> {"chips", "pending", "filters", "search": {"count", "total", "rows", "where"}, "model_ms", "db_ms"}
"exclude" maps a dismissed filter to the value it had, so it stays off until the value changes.
"""

import argparse
import atexit
import json
import subprocess
import time
import urllib.request
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from poc.search.filters import EAST, NONE, NOT_MENTIONED, WEST

from . import data
from .bench import head_engine, jev_engine
from .filters import QUESTIONS, to_filters

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HEADS = HERE / "heads"
LAYA_CHECKPOINT = HERE / "laya" / "checkpoint"
LAYA_PYTHON = ROOT / ".vendor" / "laya" / ".venv" / "bin" / "python"
LAYA_URL = "http://127.0.0.1:8095"
STATE_NAMES = {**EAST, **WEST}

CATEGORY = {"gas_transport": "Gas & transport", "grocery": "Groceries", "shopping": "Shopping",
            "food_dining": "Dining", "entertainment": "Entertainment", "health_fitness": "Health & fitness",
            "home": "Home", "kids_pets": "Kids & pets", "personal_care": "Personal care", "travel": "Travel"}
CHANNEL = {"online": "Online", "in_person": "In person"}
FRAUD = {"fraud": "Fraud", "legit": "Not fraud"}
AMOUNT_KIND = {"over": "Over $…", "under": "Under $…", "between": "Between $… and $…", "around": "Around $…",
               "large": "Large purchases", "small": "Small purchases"}
DATE_KIND = {"month": "In a month…", "weekday": "On a weekday…", "weekend": "Weekends", "today": "Today",
             "yesterday": "Yesterday", "past_week": "Past week", "this_month": "This month",
             "last_month": "Last month"}
DAYS = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]


def money(n):
    return f"${n:,.0f}" if n == int(n) else f"${n:,.2f}"


def amount_label(rng):
    lo, hi = rng
    if hi is None:
        return f"Over {money(lo)}"
    if lo is None:
        return f"Up to {money(hi)}"
    return f"{money(lo)}–{money(hi)}"


def date_label(value):
    if value.startswith("dow:"):
        days = [int(d) for d in value[4:].split(",")]
        return "Weekends" if days == [5, 6] else " & ".join(DAYS[d] for d in days)
    a, b = (date.fromisoformat(x) for x in value.split(".."))
    if a == b:
        return a.strftime("%b %-d")
    return f"{a.strftime('%b %-d')}–{b.strftime('%-d') if a.month == b.month else b.strftime('%b %-d')}"


def chips(filters, answers):
    def conf(key):
        return round(answers[key]["confidence"], 3)

    out = []
    if filters["category"]:
        out.append({"field": "category", "value": filters["category"], "label": CATEGORY[filters["category"]],
                    "confidence": conf("category")})
    if filters["channel"]:
        out.append({"field": "channel", "value": filters["channel"], "label": CHANNEL[filters["channel"]],
                    "confidence": conf("channel")})
    if filters["amount_range"]:
        out.append({"field": "amount", "value": filters["amount_range"], "label": amount_label(filters["amount_range"]),
                    "confidence": conf("amount")})
    if filters["date_value"]:
        out.append({"field": "date", "value": filters["date_value"], "label": date_label(filters["date_value"]),
                    "confidence": conf("date")})
    if filters["state"]:
        key = "state_east" if filters["state"] in EAST else "state_west"
        out.append({"field": "state", "value": filters["state"], "label": STATE_NAMES[filters["state"]],
                    "confidence": conf(key)})
    if filters["fraud"]:
        out.append({"field": "fraud", "value": filters["fraud"], "label": FRAUD[filters["fraud"]],
                    "confidence": conf("fraud")})
    return out


def pending(filters, answers):
    """What the engine leans towards but hasn't applied: not confident enough, or no number yet."""
    out = []
    for key, a in answers.items():
        choice = a["choice"]
        if choice in (NONE, NOT_MENTIONED):
            continue
        if key.startswith("state"):
            applied, label = filters["state"] == choice, STATE_NAMES[choice]
        elif key == "amount":
            applied, label = filters["amount"] == choice, AMOUNT_KIND[choice]
        elif key == "date":
            applied, label = filters["date"] == choice, DATE_KIND[choice]
        else:
            labels = {"category": CATEGORY, "channel": CHANNEL, "fraud": FRAUD}[key]
            applied, label = filters[key] == choice, labels[choice]
        if not applied:
            out.append({"label": label, "confidence": round(a["confidence"], 3), "abstain": a["abstain"]})
    return out


def start_laya():
    """Start laya_server.py in Laya's environment and wait until it answers; None if Laya isn't set up."""
    if not (LAYA_PYTHON.exists() and LAYA_CHECKPOINT.exists()):
        return None
    try:
        urllib.request.urlopen(LAYA_URL + "/health", timeout=1)
        return "already running"
    except OSError:
        pass
    proc = subprocess.Popen([str(LAYA_PYTHON), "-W", "ignore", "-m", "poc.payments.laya_server",
                             "--checkpoint", str(LAYA_CHECKPOINT)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    for _ in range(240):
        time.sleep(0.5)
        if proc.poll() is not None:
            return None
        try:
            urllib.request.urlopen(LAYA_URL + "/health", timeout=1)
            return proc
        except OSError:
            continue
    return None


def ask_laya(text):
    req = urllib.request.Request(LAYA_URL + "/answer", json.dumps({"q": text}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    if "error" in body:
        raise RuntimeError(body["error"])
    return {"answers": body["answers"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--no-jev", action="store_true", help="skip Jev (no network calls)")
    ap.add_argument("--no-laya", action="store_true")
    args = ap.parse_args()

    db = data.connect()
    total = db.execute("SELECT COUNT(*) FROM txn").fetchone()[0]
    engines, labels = {}, {}
    print("loading the Gemma head ...", flush=True)
    _, head_ask = head_engine(HEADS if HEADS.exists() else None)
    head_lock = Lock()

    def ask_head(text):
        with head_lock:  # MLX runs one request at a time
            return head_ask(text)

    engines["head"] = ask_head
    labels["head"] = "Gemma head · " + ("trained" if HEADS.exists() else "no examples")
    if not args.no_laya:
        print("starting Laya ...", flush=True)
        if start_laya():
            engines["laya"] = ask_laya
            labels["laya"] = "Laya · trained"
    if not args.no_jev:
        try:
            _, jev_ask = jev_engine()
            engines["jev"] = jev_ask
            labels["jev"] = "Jev · hosted, untrained"
        except SystemExit as e:
            print(f"no Jev: {e}", flush=True)
    db_lock = Lock()

    def run_search(filters, exclude):
        filters = dict(filters)
        values = {"category": "category", "channel": "channel", "amount": "amount_range", "date": "date_value",
                  "state": "state", "fraud": "fraud"}
        for field, value in exclude.items():
            key = values.get(field)
            if key and filters.get(key) is not None and filters[key] == value:
                filters[key] = None
                if field == "amount":
                    filters["amount"] = None
                if field == "date":
                    filters["date"] = None
        t = time.perf_counter()
        with db_lock:
            result = data.search(db, filters)
        shown, params = result["where"], list(result["params"])
        while "?" in shown and params:  # for display only; the query itself is parameterised
            p = params.pop(0)
            shown = shown.replace("?", repr(p) if isinstance(p, str) else str(p), 1)
        result.update(where=shown, total=total)
        del result["params"]
        return filters, result, (time.perf_counter() - t) * 1000

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/health":
                return self._send(200, {"engines": labels, "threshold": 0.5, "total": total})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/search":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                q, engine = str(req.get("q", "")).strip(), req.get("engine", "head")
                if engine not in engines:
                    return self._send(400, {"error": f"engine {engine!r} is not available"})
                body = {"engine": engine, "q": q, "chips": [], "pending": []}
                filters = {k: None for k in ("category", "channel", "amount", "amount_range", "date",
                                             "date_value", "state", "fraud")}
                model_ms = 0.0
                if q:
                    t = time.perf_counter()
                    answers = engines[engine](q)["answers"]
                    model_ms = (time.perf_counter() - t) * 1000
                    filters = to_filters(q, answers, float(req.get("threshold", 0.5)))
                applied, result, db_ms = run_search(filters, dict(req.get("exclude") or {}))
                if q:
                    body.update(chips=chips(applied, answers), pending=pending(filters, answers))
                body.update(filters=applied, search=result, model_ms=round(model_ms, 1), db_ms=round(db_ms, 2))
                self._send(200, body)
            except Exception as e:  # a demo server: report the error to the page
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"payments search on http://127.0.0.1:{args.port} with {', '.join(engines)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
