"""Phase 0 of the search proof of concept: is the decision head fast and steady enough to turn a
search box into filters as someone types?

    python -m poc.search.measure                  # every query in queries.txt
    python -m poc.search.measure --no-debias      # listed option order only (half the passes)
    python -m poc.search.measure --no-prior       # skip the blank-input correction
    python -m poc.search.measure --limit 10       # a quick look
    python -m poc.search.measure --report results/search-phase0-....json   # re-read a run

Each query is typed a word at a time, and every prefix is sent through all the filter questions,
as a search box would on each finished word. For each run it reports
  - accuracy on the finished query, per filter and for the whole filter set, at a few
    confidence thresholds (the threshold decides when a filter is applied);
  - wrong filters on the way: on a partly typed query, a filter applied with a value that is
    neither what has been typed so far nor where the query ends up ("new" -> New Jersey);
  - the head's own accuracy per question before any threshold, to tell the head from the gate;
  - latency per update (all questions for one prefix), p50 and p95.
Raw answers are saved, so thresholds can be re-scored with --report without the model.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .filters import EAST, NONE, NOT_MENTIONED, QUESTIONS, to_filters, where_clause

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIELDS = ("state", "kids", "age", "age_range", "marital")
CHIPS = ("state", "kids", "age", "marital")  # age_range rides with age
THRESHOLDS = (0.5, 0.7, 0.9)


# --- queries --------------------------------------------------------------------------

def _range(text):
    lo, hi = text.split("-")
    return [int(lo) if lo else None, int(hi) if hi else None]


def parse_queries(path):
    """Queries from the segment format described at the top of queries.txt."""
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
            for kv in filter(None, (m.group(2) or "").split(",")):
                key, value = kv.split("=", 1)
                if key == "age":
                    kind, rng = value.split(" ")
                    labels["age"], labels["age_range"] = kind, _range(rng)
                else:
                    labels[key] = value
            segments.append((m.group(1), labels))
        queries.append({"id": f"{lang}{len(queries) + 1:03d}", "lang": lang, "segments": segments,
                        "text": " ".join(t for t, _ in segments)})
    return queries


def prefixes(query):
    """(text typed so far, filters expected so far) after each word."""
    words, ends = [], []
    for text, labels in query["segments"]:
        words += text.split()
        ends.append((len(words), labels))
    out = []
    for k in range(1, len(words) + 1):
        expected = dict.fromkeys(FIELDS)
        for end, labels in ends:
            if end <= k:
                expected.update(labels)
        out.append((" ".join(words[:k]), expected))
    return out


def head_labels(expected):
    """The option each question should pick, before any threshold."""
    state = expected["state"]
    return {
        "state_east": state if state in EAST else NONE,
        "state_west": state if state and state not in EAST else NONE,
        "kids": expected["kids"] or NOT_MENTIONED,
        "age": expected["age"] or NOT_MENTIONED,
        "marital": expected["marital"] or NOT_MENTIONED,
    }


# --- running ----------------------------------------------------------------------------

def run(queries, debias, prior=True, heads=None):
    from decision import DecisionModel

    t = time.perf_counter()
    dm = DecisionModel(debias_order=debias)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    dm.answer("warm up", QUESTIONS)  # builds the prompts, blank-input priors and prefix caches
    warm_s = time.perf_counter() - t
    if not prior:
        # The blank-input prior assumes a blank input says nothing. For search filters it says
        # "not mentioned" very confidently, so dividing it out pushes every partly typed query
        # away from "not mentioned". Measured with and without.
        for s in dm.schemas.values():
            for view in s.views:
                view.prior = None
    if heads:
        dm.load_heads(heads)  # trained by poc.search.train_heads (no prior, listed order)
    passes = sum(len(dm.schema(k, q).views) for k, q in QUESTIONS.items())
    records = []
    for i, q in enumerate(queries):
        steps = prefixes(q)
        for w, (text, expected) in enumerate(steps):
            t = time.perf_counter()
            answers = dm.answer(text, QUESTIONS)
            ms = (time.perf_counter() - t) * 1000
            records.append({"id": q["id"], "lang": q["lang"], "words": w + 1,
                            "final": w == len(steps) - 1, "text": text, "expected": expected,
                            "answers": answers, "ms": round(ms, 1)})
        print(f"[{i + 1}/{len(queries)}] {q['text']!r}  {records[-1]['ms']:.0f} ms", flush=True)
    config = {"model": dm.backbone.name, "debias_order": debias, "prior": prior,
              "heads": str(heads) if heads else None,
              "passes_per_update": passes,
              "load_s": round(load_s, 1), "warm_s": round(warm_s, 1), "queries": len(queries),
              "updates": len(records)}
    return config, records


# --- scoring ----------------------------------------------------------------------------

def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def score(records, threshold):
    finals = [r for r in records if r["final"]]
    final_by_id = {r["id"]: r["expected"] for r in finals}
    field_ok = {f: 0 for f in FIELDS}
    exact, exact_by_lang, n_by_lang, failures = 0, {}, {}, []
    for r in finals:
        got = to_filters(r["text"], r["answers"], threshold)
        ok = {f: got[f] == r["expected"][f] for f in FIELDS}
        for f in FIELDS:
            field_ok[f] += ok[f]
        n_by_lang[r["lang"]] = n_by_lang.get(r["lang"], 0) + 1
        if all(ok.values()):
            exact += 1
            exact_by_lang[r["lang"]] = exact_by_lang.get(r["lang"], 0) + 1
        else:
            failures.append((r, got))
    # Partly typed queries: every prefix but the last.
    partial = [r for r in records if not r["final"]]
    wrong_prefixes, wrong_chips, early, waiting, missed = 0, 0, 0, 0, 0
    wrong_examples = []
    for r in partial:
        got = to_filters(r["text"], r["answers"], threshold)
        final = final_by_id[r["id"]]
        bad = []
        for f in CHIPS:
            g, e = got[f], r["expected"][f]
            if g is None:
                missed += e is not None
                continue
            if g == e:
                continue
            if e is None and g == final[f]:
                early += 1  # right, just sooner than the segment ended
                continue
            bad.append(f"{f}={g}")
        waiting += sum(r["expected"][f] is not None for f in CHIPS)
        if bad:
            wrong_prefixes += 1
            wrong_chips += len(bad)
            wrong_examples.append((r["text"], bad))
    return {
        "n": len(finals), "field_ok": field_ok, "exact": exact, "exact_by_lang": exact_by_lang,
        "n_by_lang": n_by_lang, "failures": failures, "partial": len(partial),
        "wrong_prefixes": wrong_prefixes, "wrong_chips": wrong_chips, "early": early,
        "missed": missed, "waiting": waiting, "wrong_examples": wrong_examples,
    }


def head_accuracy(records):
    """Argmax accuracy per question on finished queries, and on partly typed ones."""
    out = {}
    for key in QUESTIONS:
        for name, subset in (("final", [r for r in records if r["final"]]),
                             ("partial", [r for r in records if not r["final"]])):
            ok = sum(r["answers"][key]["choice"] == head_labels(r["expected"])[key] for r in subset)
            out[(key, name)] = (ok, len(subset))
    return out


def fmt_filters(f):
    parts = [f"{k}={v}" for k, v in f.items() if v is not None and k != "age_range"]
    if f.get("age_range"):
        parts.append("ages {}–{}".format(*["" if x is None else x for x in f["age_range"]]))
    return ", ".join(parts) or "(none)"


def report(config, records):
    ms = np.array([r["ms"] for r in records])
    lines = [
        f"# Search POC phase 0: {'order-averaged' if config['debias_order'] else 'listed order only'}, "
        f"{'with' if config.get('prior', True) else 'without'} the blank-input prior"
        f"{', trained heads' if config.get('heads') else ', zero labels'}",
        "",
        f"Model `{config['model']}`, {config['queries']} queries typed word by word = "
        f"{config['updates']} updates, {config['passes_per_update']} forward passes per update. "
        f"Model load {config['load_s']} s; first query (prompts, priors, prefix caches) {config['warm_s']} s.",
        "",
        "## Latency per update (all filter questions for one prefix)",
        "",
        "| p50 | p95 | mean | max |",
        "|---|---|---|---|",
        f"| {np.percentile(ms, 50):.0f} ms | {np.percentile(ms, 95):.0f} ms | {ms.mean():.0f} ms | {ms.max():.0f} ms |",
        "",
        "## The head on its own (top option, no threshold)",
        "",
        "| Question | Finished query | Partly typed |",
        "|---|---|---|",
    ]
    acc = head_accuracy(records)
    for key in QUESTIONS:
        f_ok, f_n = acc[(key, "final")]
        p_ok, p_n = acc[(key, "partial")]
        lines.append(f"| {key} | {pct(f_ok, f_n)} | {pct(p_ok, p_n)} |")
    scores = {t: score(records, t) for t in THRESHOLDS}
    lines += [
        "",
        "## Applied filters by confidence threshold",
        "",
        "A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and "
        "its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes "
        "with a filter applied that is neither right yet nor where the query ends up.",
        "",
        "| Threshold | All filters right | state | kids | age kind | age range | marital "
        "| Wrong on the way (prefixes) | Applied early but right | Missed while typing |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, s in scores.items():
        n = s["n"]
        fo = s["field_ok"]
        lines.append(
            f"| {t} | {pct(s['exact'], n)} | {pct(fo['state'], n)} | {pct(fo['kids'], n)} "
            f"| {pct(fo['age'], n)} | {pct(fo['age_range'], n)} | {pct(fo['marital'], n)} "
            f"| {s['wrong_prefixes']} of {s['partial']} ({pct(s['wrong_prefixes'], s['partial'])}) "
            f"| {s['early']} | {s['missed']} of {s['waiting']} |")
    lines += ["", "All filters right, by language:", "", "| Threshold | " +
              " | ".join(scores[THRESHOLDS[0]]["n_by_lang"]) + " |",
              "|---|" + "---|" * len(scores[THRESHOLDS[0]]["n_by_lang"])]
    for t, s in scores.items():
        lines.append(f"| {t} | " + " | ".join(
            f"{pct(s['exact_by_lang'].get(lang, 0), n)} ({n})" for lang, n in s["n_by_lang"].items()) + " |")
    t = THRESHOLDS[0]
    s = scores[t]
    lines += ["", f"## Finished queries with a wrong filter (threshold {t})", "",
              "| Query | Expected | Got | SQL |", "|---|---|---|---|"]
    for r, got in s["failures"]:
        lines.append(f"| {r['text']} | {fmt_filters(r['expected'])} | {fmt_filters(got)} "
                     f"| `{where_clause(got)}` |")
    lines += ["", f"## Wrong filters while typing (threshold {t})", "", "| Typed so far | Wrong filter |",
              "|---|---|"]
    for text, bad in s["wrong_examples"]:
        lines.append(f"| {text} | {', '.join(bad)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", default=HERE / "queries.txt")
    ap.add_argument("--no-debias", action="store_true", help="listed option order only")
    ap.add_argument("--no-prior", action="store_true", help="skip the blank-input correction")
    ap.add_argument("--heads", help="folder of trained heads (from poc.search.train_heads)")
    ap.add_argument("--limit", type=int, help="only the first N queries")
    ap.add_argument("--report", help="re-score a saved run instead of running the model")
    args = ap.parse_args()

    if args.report:
        saved = json.loads(Path(args.report).read_text())
        print(report(saved["config"], saved["records"]))
        return
    if args.heads and not (args.no_prior and args.no_debias):
        ap.error("trained heads are for listed order without the prior: add --no-debias --no-prior")
    queries = parse_queries(args.queries)[: args.limit]
    config, records = run(queries, debias=not args.no_debias, prior=not args.no_prior, heads=args.heads)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = ("debias" if config["debias_order"] else "listed") + ("" if config["prior"] else "-noprior") \
        + ("-trained" if config["heads"] else "")
    out = ROOT / "results" / f"search-phase0-{stamp}-{tag}"
    out.with_suffix(".json").write_text(json.dumps({"config": config, "records": records}, indent=1))
    text = report(config, records)
    out.with_suffix(".md").write_text(text)
    print(text)
    print(f"saved {out}.json and .md")


if __name__ == "__main__":
    main()
