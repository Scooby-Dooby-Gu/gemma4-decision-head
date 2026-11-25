"""Card transaction search: the same test as poc/search, on payments. Runs one engine over every
word-by-word prefix of queries.txt, and compares saved runs.

    python -m poc.payments.bench head                         # Gemma head, no examples
    python -m poc.payments.bench head --heads poc/payments/heads
    python -m poc.payments.bench llm                          # Gemma 4 12B through Ollama
    python -m poc.payments.bench jev                          # Jev (key in .env.local)
    python -m poc.payments.bench compare results/payments-head-*.json results/payments-llm-*.json ...

Every run is saved to results/payments-<engine>-<stamp>.json in one record format, and scored
the same way: a finished query is right when every filter matches, and "wrong while typing" is a
half-typed query with a filter that is neither right yet nor what the finished query needs.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from poc.search.filters import EAST, NONE, NOT_MENTIONED, WEST
from poc.search.jev import ENDPOINTS, load_key, name_keys

from .filters import CATEGORIES, FIELDS, QUESTIONS, TODAY, WEEKDAYS, to_filters

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "results"
THRESHOLDS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)

# --- queries --------------------------------------------------------------------------------


def _range(text):
    lo, hi = text.split("-")
    num = lambda s: (float(s) if "." in s else int(s)) if s else None
    return [num(lo), num(hi)]


def parse_queries(path=HERE / "queries.txt"):
    queries, lang = [], "en"
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            lang = line[1:]
            continue
        segments = []
        for raw in line.split(" | "):
            m = re.fullmatch(r"(.*?)\s*(?:\{(.*)\})?", raw.strip())
            labels = {}
            for kv in filter(None, (m.group(2) or "").split(";")):
                key, value = (s.strip() for s in kv.split("=", 1))
                if key == "amount":
                    kind, rng = value.split(" ")
                    labels["amount"], labels["amount_range"] = kind, _range(rng)
                elif key == "date":
                    kind, val = value.split(" ")
                    labels["date"], labels["date_value"] = kind, val
                else:
                    labels[key] = value
            segments.append((m.group(1), labels))
        queries.append({"id": f"{lang}{len(queries) + 1:03d}", "lang": lang, "segments": segments,
                        "text": " ".join(t for t, _ in segments)})
    return queries


EMPTY = dict.fromkeys(("category", "channel", "amount", "amount_range", "date", "date_value", "state", "fraud"))


def prefixes(query):
    """(text typed so far, filters expected so far) after each word."""
    words, ends = [], []
    for text, labels in query["segments"]:
        words += text.split()
        ends.append((len(words), labels))
    out = []
    for k in range(1, len(words) + 1):
        expected = dict(EMPTY)
        for end, labels in ends:
            if end <= k:
                expected.update(labels)
        out.append((" ".join(words[:k]), expected))
    return out


def head_labels(expected):
    """The option each question should pick."""
    s = expected["state"]
    return {"category": expected["category"] or NOT_MENTIONED, "channel": expected["channel"] or NOT_MENTIONED,
            "amount": expected["amount"] or NOT_MENTIONED, "date": expected["date"] or NOT_MENTIONED,
            "state_east": s if s in EAST else NONE, "state_west": s if s and s not in EAST else NONE,
            "fraud": expected["fraud"] or NOT_MENTIONED}


def run(queries, ask):
    """ask(text) -> (record fields); one record per typed prefix."""
    records = []
    for i, q in enumerate(queries):
        steps = prefixes(q)
        for w, (text, expected) in enumerate(steps):
            t = time.perf_counter()
            out = ask(text)
            ms = (time.perf_counter() - t) * 1000
            records.append({"id": q["id"], "lang": q["lang"], "final": w == len(steps) - 1, "text": text,
                            "expected": expected, "ms": round(ms, 1), **out})
        print(f"[{i + 1}/{len(queries)}] {q['text']!r}  {records[-1]['ms']:.0f} ms", flush=True)
    return records


# --- engines --------------------------------------------------------------------------------

def head_engine(heads=None):
    from decision import DecisionModel

    dm = DecisionModel(debias_order=False)  # listed order, no blank-input prior: the search setup
    dm.answer("warm up", QUESTIONS)
    for s in dm.schemas.values():
        for view in s.views:
            view.prior = None
    if heads:
        dm.load_heads(heads)
    config = {"engine": "head", "model": dm.backbone.name, "heads": str(heads) if heads else None,
              "passes_per_update": sum(len(dm.schema(k, q).views) for k, q in QUESTIONS.items())}
    return config, lambda text: {"answers": dm.answer(text, QUESTIONS)}


OLLAMA = "http://127.0.0.1:11434/api/chat"
LLM_MODEL = "gemma4:12b-mlx"
DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
LLM_SYSTEM = (
    "You turn a search typed over credit card transactions into filters. The query may be unfinished, "
    "because it is sent as the person types, and may be in English, Spanish or French. Fill a field only "
    "when the query asks for it; otherwise use null. Today is "
    f"{TODAY.isoformat()} ({TODAY.strftime('%A')}).\n"
    "- category: one of " + ", ".join(f"{k} ({v})" for k, v in CATEGORIES.items()) + ".\n"
    "- channel: online or in_person.\n"
    "- amount_min / amount_max: dollars, inclusive. \"over $100\" is amount_min 100; \"under $20\" is "
    "amount_max 20; \"around $100\" is 90 to 110 (10% either side); large purchases are amount_min 500; "
    "small purchases are amount_max 20.\n"
    "- date_from / date_to: YYYY-MM-DD, inclusive. A named month is that whole month this year (last year "
    "if it hasn't come yet); \"last week\" or \"past week\" is the last seven days including today; "
    "\"this month\" is from the 1st to today; \"last month\" is the previous calendar month.\n"
    "- weekdays: a list of day names (monday ... sunday) for a day-of-week filter such as \"on fridays\"; "
    "weekends are [\"saturday\", \"sunday\"]. Use weekdays instead of dates for these.\n"
    "- state: the two-letter US state code where the cardholders live.\n"
    "- fraud: fraud (fraudulent, suspicious or flagged) or legit (legitimate only).\n"
    "Reply with only a JSON object with exactly these keys: "
    '{"category": ..., "channel": ..., "amount_min": ..., "amount_max": ..., "date_from": ..., '
    '"date_to": ..., "weekdays": ..., "state": ..., "fraud": ...}'
)


def llm_filters(out):
    """The 12B's JSON as the same filter dict to_filters returns."""
    num = lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    lo, hi = num(out.get("amount_min")), num(out.get("amount_max"))
    days = out.get("weekdays") or []
    days = sorted({DAY_NAMES.index(d.lower()) for d in days if isinstance(d, str) and d.lower() in DAY_NAMES})
    when = None
    if days:
        when = "dow:" + ",".join(map(str, days))
    elif out.get("date_from") or out.get("date_to"):
        when = f"{out.get('date_from') or out.get('date_to')}..{out.get('date_to') or TODAY.isoformat()}"
    pick = lambda v, allowed: v if v in allowed else None
    return {
        "category": pick(out.get("category"), CATEGORIES),
        "channel": pick(out.get("channel"), ("online", "in_person")),
        "amount": "range" if lo is not None or hi is not None else None,
        "amount_range": [lo, hi] if lo is not None or hi is not None else None,
        "date": "range" if when else None,
        "date_value": when,
        "state": pick(out.get("state"), {**EAST, **WEST}),
        "fraud": pick(out.get("fraud"), ("fraud", "legit")),
    }


def llm_engine():
    def ask(text):
        body = {"model": LLM_MODEL, "stream": False, "think": False, "keep_alive": "30m",
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": text}]}
        req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())["message"]["content"]
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            out = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            out = {}
        return {"filters": llm_filters(out), "raw": raw}

    ask("warm up")
    return {"engine": "llm", "model": f"{LLM_MODEL} (Ollama)"}, ask


def jev_engine(model=None):
    provider, key = load_key()
    _, url, default = ENDPOINTS[provider]
    model = model or default
    questions, back = name_keys(QUESTIONS)  # states keyed by name: "in" isn't Indiana

    def ask(text):
        body = {"model": model, "state": text, "questions": questions}
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 524, 529) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"HTTP {e.code} from {url}: {e.read()[:500].decode(errors='replace')}")
        got = raw.get("answers", raw)
        answers = {}
        for name, q in QUESTIONS.items():
            a = got.get(name) or {}
            choice = back.get(name, {}).get(a.get("choice"), a.get("choice"))
            if a.get("type") == "choice" and choice in q["criteria"]:
                answers[name] = {"choice": choice, "confidence": float(a.get("confidence") or 0.0), "abstain": False}
            else:
                answers[name] = {"choice": next(iter(q["criteria"])), "confidence": 0.0, "abstain": True}
        return {"answers": answers, "usage": raw.get("usage", {}), "served": raw.get("model")}

    ask("warm up")
    return {"engine": "jev", "model": model, "provider": provider,
            "note": "ms is end to end from this machine, network included"}, ask


def laya_engine(default="multilingual", checkpoint=None):
    """Laya's encoder checkpoints (English, and multilingual for other languages, picked by
    Laya's own router), run locally. Needs Laya's own environment:

        .vendor/laya/.venv/bin/python -m poc.payments.bench laya

    The state questions have 25 to 28 options, past the ~20 Laya's docs suggest at its default
    192-token question budget, so the budget is raised to 256 (the state still gets 248 tokens).
    Confidence is the chosen option's probability, gated like the other decision models; Laya's
    own abstain flag and route are kept in the record."""
    import laya
    from laya.router import Router

    if checkpoint:  # one fine-tuned checkpoint (poc.payments.laya_train) answers every language
        agent = laya.load(checkpoint)
        predict = lambda text, qs: dict(agent.predict(text, qs), routing={"model": "fine-tuned"})
    else:
        # Short text whose language can't be told goes to `default`: Laya ships "multilingual".
        router = Router(vega=False, preload=True, default=default)
        predict = lambda text, qs: router.predict(text, qs, head_max_len=256)
    questions, back = name_keys(QUESTIONS)

    def ask(text):
        out = predict(text, questions)
        answers, extra = {}, {}
        for name, q in QUESTIONS.items():
            a = out["answers"][name]
            choice = back.get(name, {}).get(a.get("choice"), a.get("choice"))
            probs = a.get("probabilities") or {}
            answers[name] = {"choice": choice, "confidence": float(probs.get(a.get("choice"), 0.0)), "abstain": False}
            extra[name] = {"laya_abstain": a.get("abstain"), "route": a.get("route")}
        return {"answers": answers, "laya": extra, "checkpoint": (out.get("routing") or {}).get("model")}

    ask("warm up")
    ask("calentamiento")  # loads the multilingual checkpoint too

    return {"engine": "laya", "default": default, "checkpoint": str(checkpoint) if checkpoint else None,
            "model": f"laya {laya.__version__} encoders (english + multilingual), local"}, ask


# --- scoring -------------------------------------------------------------------------------

def filters_of(record, threshold):
    if "filters" in record:
        return record["filters"]
    return to_filters(record["text"], record["answers"], threshold)


def score(records, threshold=0.5):
    finals = [r for r in records if r["final"]]
    final_by_id = {r["id"]: r["expected"] for r in finals}
    field_ok = dict.fromkeys(FIELDS, 0)
    exact, by_lang, n_lang, misses = 0, {}, {}, []
    for r in finals:
        got = filters_of(r, threshold)
        ok = {f: got[f] == r["expected"][f] for f in FIELDS}
        for f in FIELDS:
            field_ok[f] += ok[f]
        n_lang[r["lang"]] = n_lang.get(r["lang"], 0) + 1
        if all(ok.values()):
            exact += 1
            by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
        else:
            misses.append((r["text"], {f: r["expected"][f] for f in FIELDS if not ok[f]},
                           {f: got[f] for f in FIELDS if not ok[f]}))
    partial = [r for r in records if not r["final"]]
    wrong, wrong_examples = 0, []
    for r in partial:
        got, final = filters_of(r, threshold), final_by_id[r["id"]]
        bad = [f"{f}={got[f]}" for f in FIELDS
               if got[f] is not None and got[f] != r["expected"][f] and got[f] != final[f]]
        if bad:
            wrong += 1
            wrong_examples.append((r["text"], bad))
    ms = np.array([r["ms"] for r in records])
    return {"n": len(finals), "exact": exact, "field_ok": field_ok, "by_lang": by_lang, "n_lang": n_lang,
            "partial": len(partial), "wrong": wrong, "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)), "misses": misses, "wrong_examples": wrong_examples}


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def label(config):
    if config["engine"] == "head":
        return "Gemma head, " + ("trained" if config.get("heads") else "no examples")
    if config["engine"] == "laya":
        if config.get("checkpoint"):
            return "Laya, trained"
        return "Laya" + (", English fallback" if config.get("default") == "english" else "")
    return {"llm": "Gemma 4 12B writing JSON", "jev": "Jev"}[config["engine"]]


def compare(paths, thresholds):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    names = [label(r["config"]) for r in runs]
    scores = [score(r["records"], t) for r, t in zip(runs, thresholds)]
    first = scores[0]
    lines = ["# Card transaction search: " + " vs ".join(names), "",
             f"{first['n']} queries typed word by word ({first['n'] + first['partial']} updates). "
             "Thresholds: " + ", ".join(f"{n} {t}" for n, t, r in zip(names, thresholds, runs)
                                        if "answers" in r["records"][0]) + ". Runs: "
             + ", ".join(f"`{Path(p).name}`" for p in paths) + ".", "",
             "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]

    def row(name, cell):
        lines.append(f"| {name} | " + " | ".join(cell(s) for s in scores) + " |")

    row("Time per update, p50 / p95", lambda s: f"{s['p50']:.0f} / {s['p95']:.0f} ms")
    row("Finished queries with every filter right", lambda s: pct(s["exact"], s["n"]))
    for f in FIELDS:
        row(f, lambda s, f=f: pct(s["field_ok"][f], s["n"]))
    for lang, n in first["n_lang"].items():
        row(f"Every filter right, {lang} ({n})", lambda s, lang=lang, n=n: pct(s["by_lang"].get(lang, 0), n))
    row(f"Wrong filter while typing (of {first['partial']} half-typed)", lambda s: pct(s["wrong"], s["partial"]))
    for name, r in zip(names, runs):
        if "answers" not in r["records"][0]:
            continue
        lines += ["", f"## {name} by threshold", "", "| Threshold | Every filter right | Wrong while typing |",
                  "|---|---|---|"]
        for t in THRESHOLDS:
            s = score(r["records"], t)
            lines.append(f"| {t} | {pct(s['exact'], s['n'])} | {pct(s['wrong'], s['partial'])} |")
    for name, s in zip(names, scores):
        lines += ["", f"## {name}: finished queries with a wrong filter", "", "| Query | Expected | Got |",
                  "|---|---|---|"]
        lines += [f"| {text} | {exp} | {got} |" for text, exp, got in s["misses"]]
        lines += ["", f"## {name}: wrong filters while typing", "", "| Typed so far | Wrong filter |", "|---|---|"]
        lines += [f"| {text} | {', '.join(bad)} |" for text, bad in s["wrong_examples"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("engine", choices=("head", "llm", "jev", "laya", "laya-en", "compare"))
    ap.add_argument("runs", nargs="*", help="for compare: saved runs, in column order")
    ap.add_argument("--heads", help="trained heads folder (head engine)")
    ap.add_argument("--laya-checkpoint", help="fine-tuned Laya checkpoint (laya engine)")
    ap.add_argument("--thresholds", help="for compare: one per run, comma-separated (default 0.5)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.engine == "compare":
        th = [float(t) for t in args.thresholds.split(",")] if args.thresholds else [0.5] * len(args.runs)
        text = compare(args.runs, th)
        out = RESULTS / f"payments-compare-{stamp}.md"
        out.write_text(text)
        print(text)
        print(f"saved {out}")
        return
    config, ask = {"head": lambda: head_engine(args.heads), "llm": llm_engine, "jev": jev_engine,
                   "laya": lambda: laya_engine(checkpoint=args.laya_checkpoint), "laya-en": lambda: laya_engine("english")}[args.engine]()
    queries = parse_queries()[: args.limit]
    records = run(queries, ask)
    tag = args.engine + ("-trained" if args.heads or args.laya_checkpoint else "")
    out = RESULTS / f"payments-{tag}-{stamp}.json"
    config.update(queries=len(queries), updates=len(records))
    out.write_text(json.dumps({"config": config, "records": records}, indent=1, ensure_ascii=False))
    s = score(records)
    print(f"every filter right {pct(s['exact'], s['n'])}, wrong while typing {pct(s['wrong'], s['partial'])}, "
          f"p50 {s['p50']:.0f} ms; saved {out}")


if __name__ == "__main__":
    main()
