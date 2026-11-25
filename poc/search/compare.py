"""Score Gemma 4 12B writing JSON filters (through Ollama) on the phase 0 queries, next to a
saved decision-head run, with the same definitions for both.

    python -m poc.search.compare results/search-phase0-...-listed-noprior.json

Ages are compared as ranges, since the 12B writes a range and never an age "kind". Everything
else follows measure.py: each query typed a word at a time, "wrong while typing" is a filter
applied on a partly typed query that is neither right yet nor where the query ends up.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .app import ask_llm
from .filters import to_filters
from .measure import HERE, ROOT, parse_queries, prefixes

FIELDS = ("state", "kids", "age_range", "marital")


def run_llm(queries):
    ask_llm("warm up")
    records = []
    for i, q in enumerate(queries):
        steps = prefixes(q)
        for w, (text, expected) in enumerate(steps):
            t = time.perf_counter()
            filters, raw = ask_llm(text)
            ms = (time.perf_counter() - t) * 1000
            records.append({"id": q["id"], "lang": q["lang"], "final": w == len(steps) - 1, "text": text,
                            "expected": expected, "filters": filters, "raw": raw, "ms": round(ms, 1)})
        print(f"[{i + 1}/{len(queries)}] {q['text']!r}  {records[-1]['ms']:.0f} ms", flush=True)
    return records


def score(records, filters_of):
    finals = [r for r in records if r["final"]]
    final_by_id = {r["id"]: r["expected"] for r in finals}
    field_ok = dict.fromkeys(FIELDS, 0)
    exact, by_lang, n_lang, misses = 0, {}, {}, []
    for r in finals:
        got = filters_of(r)
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
    wrong = 0
    for r in partial:
        got, final = filters_of(r), final_by_id[r["id"]]
        wrong += any(got[f] is not None and got[f] != r["expected"][f] and got[f] != final[f] for f in FIELDS)
    ms = np.array([r["ms"] for r in records])
    return {"n": len(finals), "exact": exact, "field_ok": field_ok, "by_lang": by_lang, "n_lang": n_lang,
            "partial": len(partial), "wrong": wrong, "p50": np.percentile(ms, 50), "p95": np.percentile(ms, 95),
            "misses": misses}


def pct(a, b):
    return f"{100 * a / b:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head_run", help="a saved measure.py run (.json) to compare against")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--llm-run", help="re-score a saved 12B run instead of calling Ollama")
    ap.add_argument("--jev-run", help="a saved poc.search.jev run, scored as a third column")
    ap.add_argument("--jev-threshold", type=float, default=0.5)
    args = ap.parse_args()

    head = json.loads(Path(args.head_run).read_text())
    if args.llm_run:
        llm_records = json.loads(Path(args.llm_run).read_text())["records"]
    else:
        llm_records = run_llm(parse_queries(HERE / "queries.txt"))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = ROOT / "results" / f"search-llm12b-{stamp}.json"
        out.write_text(json.dumps({"model": "gemma4:12b-mlx (Ollama)", "records": llm_records}, indent=1))
        print(f"saved {out}")

    h = score(head["records"], lambda r: to_filters(r["text"], r["answers"], args.threshold))
    g = score(llm_records, lambda r: r["filters"])
    engines = [("Decision head (E4B)", h), ("12B writes JSON", g)]
    intro = (f"Same {h['n']} queries, typed word by word ({h['n'] + h['partial']} updates). Head: Gemma 4 E4B, "
             f"listed order, no blank-input prior, threshold {args.threshold}, from `{Path(args.head_run).name}`. "
             "12B: `gemma4:12b-mlx` through Ollama, temperature 0, JSON asked for in the prompt (Ollama's MLX "
             "backend has no structured output). Ages compared as ranges.")
    sweep = []
    if args.jev_run:
        jev = json.loads(Path(args.jev_run).read_text())
        j = score(jev["records"], lambda r: to_filters(r["text"], r["answers"], args.jev_threshold))
        engines.append(("Jev (hosted)", j))
        cfg = jev["config"]
        intro += (f" Jev: `{cfg['model']}` via {cfg['provider']}, the head's exact questions, threshold "
                  f"{args.jev_threshold}, from `{Path(args.jev_run).name}`; its latency is end to end from "
                  "this machine, network included, while the other two run locally.")
        for t in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
            s_ = score(jev["records"], lambda r, t=t: to_filters(r["text"], r["answers"], t))
            sweep.append(f"| {t} | {pct(s_['exact'], s_['n'])} | {pct(s_['wrong'], s_['partial'])} |")
    langs = list(h["n_lang"])
    title = " vs ".join(n for n, _ in engines)
    lines = ["# Search POC: " + title, "", intro, "",
             "| | " + " | ".join(n for n, _ in engines) + " |", "|---" * (len(engines) + 1) + "|"]

    def row(label, cell):
        lines.append(f"| {label} | " + " | ".join(cell(s_) for _, s_ in engines) + " |")

    row("Latency per update, p50 / p95", lambda s_: f"{s_['p50']:.0f} / {s_['p95']:.0f} ms")
    row("All filters right (finished query)", lambda s_: pct(s_["exact"], s_["n"]))
    for f in FIELDS:
        row(f, lambda s_, f=f: pct(s_["field_ok"][f], s_["n"]))
    for lang in langs:
        row(f"All right, {lang} ({h['n_lang'][lang]})", lambda s_, lang=lang: pct(s_["by_lang"].get(lang, 0), h["n_lang"][lang]))
    row(f"Wrong filter while typing (of {h['partial']} prefixes)", lambda s_: pct(s_["wrong"], s_["partial"]))
    if sweep:
        lines += ["", "## Jev by threshold", "", "| Threshold | All right | Wrong while typing |", "|---|---|---|", *sweep]
    for name, s in [(n, s_) for n, s_ in engines][::-1]:
        lines += ["", f"## {name}: finished queries with a wrong filter", "", "| Query | Expected | Got |", "|---|---|---|"]
        for text, exp, got in s["misses"]:
            lines.append(f"| {text} | {exp} | {got} |")
    report = "\n".join(lines) + "\n"
    out = ROOT / "results" / f"search-compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    out.write_text(report)
    print(report)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
