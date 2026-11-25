"""Run the search test queries through Jev, TypeSafe's hosted decision model, word by word,
with the same filter questions the decision head gets.

    python -m poc.search.jev                    # all 134 queries, every prefix
    python -m poc.search.jev --limit 3          # a quick check first

The key comes from the environment or `.env.local` (gitignored): TYPESAFE_API_KEY calls
TypeSafe directly, otherwise OPENROUTER_API_KEY goes through OpenRouter's Decisions API.
The run is saved in measure.py's record format, so to_filters() and the compare script score
it exactly like a head run:

    python -m poc.search.compare results/<head run>.json --llm-run results/<12B run>.json \\
        --jev-run results/search-jev-<stamp>.json

Jev has no abstain: a "refusal" answer, or a failed request, counts as abstaining (no filter).
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

from .filters import QUESTIONS
from .measure import HERE, ROOT, parse_queries, prefixes

ENDPOINTS = {
    "typesafe": ("TYPESAFE_API_KEY", "https://api.typesafe.ai/v1/systemone", "jev-latest"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.14"),
}


def load_key():
    """(provider, key) from the environment, then from .env.local. The key is never printed."""
    env = dict(os.environ)
    path = ROOT / ".env.local"
    if path.exists():
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() and not name.lstrip().startswith("#"):
                env.setdefault(name.strip(), value.strip().strip("'\""))
    for provider, (var, _, _) in ENDPOINTS.items():
        if env.get(var):
            return provider, env[var]
    raise SystemExit("No key: set TYPESAFE_API_KEY or OPENROUTER_API_KEY in .env.local or the environment.")


def ask_jev(text, provider, key, model):
    """Our answer format ({"choice", "confidence", "abstain"} per question), the raw reply and its usage."""
    _, url, _ = ENDPOINTS[provider]
    body = {"model": model, "state": text, "questions": QUESTIONS}
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
    # TypeSafe answers {question: answer}; OpenRouter wraps them in {"answers": {...}}.
    got = raw.get("answers", raw)
    answers = {}
    for name, q in QUESTIONS.items():
        a = got.get(name) or {}
        if a.get("type") == "choice" and a.get("choice") in q["criteria"]:
            answers[name] = {"choice": a["choice"], "confidence": float(a.get("confidence") or 0.0),
                             "abstain": False, "probabilities": a.get("probabilities")}
        else:  # refusal or missing: no filter
            first = next(iter(q["criteria"]))
            answers[name] = {"choice": first, "confidence": 0.0, "abstain": True, "probabilities": None}
    return answers, raw.get("model"), raw.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", help="override the pinned model name")
    args = ap.parse_args()

    provider, key = load_key()
    model = args.model or ENDPOINTS[provider][2]
    queries = parse_queries(HERE / "queries.txt")[: args.limit]
    print(f"Jev via {provider}, model {model}", flush=True)
    ask_jev("warm up", provider, key, model)

    records, served, tokens = [], None, 0
    for i, q in enumerate(queries):
        steps = prefixes(q)
        for w, (text, expected) in enumerate(steps):
            t = time.perf_counter()
            answers, served, usage = ask_jev(text, provider, key, model)
            ms = (time.perf_counter() - t) * 1000
            tokens += usage.get("input_tokens", 0)
            records.append({"id": q["id"], "lang": q["lang"], "words": w + 1, "final": w == len(steps) - 1,
                            "text": text, "expected": expected, "answers": answers, "ms": round(ms, 1),
                            "usage": usage})
        print(f"[{i + 1}/{len(queries)}] {q['text']!r}  {records[-1]['ms']:.0f} ms", flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"search-jev-{stamp}.json"
    config = {"model": served or model, "provider": provider, "queries": len(queries),
              "updates": len(records), "input_tokens": tokens,
              "note": "ms is end to end from this machine, network included"}
    out.write_text(json.dumps({"config": config, "records": records}, indent=1))
    print(f"{len(records)} updates, {tokens} input tokens; saved {out}")


if __name__ == "__main__":
    main()
