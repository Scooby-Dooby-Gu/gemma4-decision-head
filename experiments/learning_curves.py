"""How many labels does the head need? Learning curves on public datasets.

    python -m experiments.learning_curves                       # every suite (~45 min first run)
    python -m experiments.learning_curves --suite clinc_banking --suite sst5
    python -m experiments.learning_curves --limit 40            # quick smoke test

For each suite, Gemma's hidden states for every train and test case are computed once and cached
in experiments/cache/ (the slow part). Then, for each size n in --sizes and each of --seeds
random draws, n training cases per option are sampled and three heads are trained and scored
on the full test set: rung 1 (bias and temperature), rung 2 (also the per-option correction),
and `train`, the guarded default (rung for the question type, checked and calibrated on a
held-out fifth). n = 0 is the zero-data head. Training takes seconds, so trying head variants is cheap once the
cache exists.

Results: results/learning-curves-<time>.json (every run) and .md (the summary tables).
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision import DecisionModel  # noqa: E402
from decision.head import Head, train  # noqa: E402
from evaluate import metrics  # noqa: E402
from experiments import suites  # noqa: E402

CACHE = ROOT / "experiments" / "cache"
SIZES = [0, 5, 10, 20, 50, 100]
OOS = "out_of_scope"
GUARDED = 3  # runs from `train`, the guarded default


def features(dm, suite, split, chunk=100):
    """(H, y) for one split, from the cache or computed once with progress."""
    (key, q), = suite["questions"].items()
    s = dm.schema(key, q)
    cases = suite[split]
    digest = hashlib.sha256(json.dumps([dm.backbone.name, [(c["state"], c["expected"]) for c in cases]])
                            .encode()).hexdigest()[:12]
    path = CACHE / f"{suite['name']}-{split}-{s.fingerprint}-{digest}.npz"
    if path.exists():
        d = np.load(path)
        return [d[f"h{v}"] for v in range(len(s.views))], d["y"]
    print(f"  extracting {split}: {len(cases)} cases x {len(s.views)} views", flush=True)
    t0, parts = time.perf_counter(), []
    for i in range(0, len(cases), chunk):
        parts.append(dm.features_all(cases[i:i + chunk], suite["questions"])[key])
        done = min(i + chunk, len(cases))
        rate = (time.perf_counter() - t0) / done
        print(f"    {done}/{len(cases)}  {rate * 1000:.0f} ms/case, {rate * (len(cases) - done) / 60:.1f} min left",
              flush=True)
    H = [np.concatenate([p[0][v] for p in parts]) for v in range(len(s.views))]
    y = np.concatenate([p[1] for p in parts])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, y=y, **{f"h{v}": h for v, h in enumerate(H)})
    return H, y


def sample(y, n, rng):
    """Indices of up to n training cases per option."""
    idx = []
    for k in np.unique(y):
        pool = np.flatnonzero(y == k)
        idx.extend(rng.choice(pool, size=min(n, len(pool)), replace=False))
    return np.array(sorted(idx))


def score(p, y, s, margin):
    m = metrics(p, y, margin)
    if OOS in s.options:  # out-of-scope as an option: how well is it caught, and how often wrongly?
        k = s.options.index(OOS)
        pred = p.argmax(1)
        m["oos_recall"] = float((pred[y == k] == k).mean()) if (y == k).any() else None
        m["oos_precision"] = float((y[pred == k] == k).mean()) if (pred == k).any() else None
        m["in_scope_accuracy"] = float((pred[y != k] == y[y != k]).mean()) if (y != k).any() else None
    return m


def run_suite(dm, name, sizes, seeds, limit):
    suite = suites.build(name)
    if limit:
        suite["train"], suite["test"] = suite["train"][:limit], suite["test"][:limit]
    (key, q), = suite["questions"].items()
    s = dm.schema(key, q)
    print(f"\n== {name}: {len(s.options)} options, {len(suite['train'])} train, {len(suite['test'])} test", flush=True)
    Htr, ytr = features(dm, suite, "train")
    Hte, yte = features(dm, suite, "test")
    D, softcap = Htr[0].shape[-1], dm.backbone.softcap
    available = int(np.bincount(ytr, minlength=len(s.options)).min())
    runs = [{"n": 0, "rung": 0, "seed": 0, "labels": 0,
             **score(dm.heads[(key, s.fingerprint)].predict(s, Hte), yte, s, dm.abstain_margin)}]
    for n in [n for n in sizes if n > 0]:
        for seed in range(seeds):
            idx = sample(ytr, n, np.random.default_rng(seed))
            Hn, yn = [h[idx] for h in Htr], ytr[idx]
            for rung in (1, 2):
                head = Head(len(s.options), D, softcap).fit(s, Hn, yn, rung=rung)
                runs.append({"n": n, "rung": rung, "seed": seed, "labels": int(len(idx)),
                             **score(head.predict(s, Hte), yte, s, dm.abstain_margin)})
            head = train(s, Hn, yn, D, softcap, seed=seed)
            runs.append({"n": n, "rung": GUARDED, "seed": seed, "labels": int(len(idx)), "chosen": head.rung,
                         **score(head.predict(s, Hte), yte, s, dm.abstain_margin)})
        print(f"  n={n:<3} done", flush=True)
    return {"name": name, "type": q["type"], "options": len(s.options), "train": len(suite["train"]),
            "test": len(suite["test"]), "min_train_per_option": available, "runs": runs}


def summarize(result):
    """Markdown table: mean (sd) over seeds for each size and rung."""
    has_oos = "oos_recall" in result["runs"][0]
    cols = ["accuracy", "ece", "abstain_rate", "accuracy_when_answering"] + (["oos_recall", "oos_precision"] if has_oos else [])
    lines = [f"### {result['name']} ({result['type']}, {result['options']} options, {result['test']} test cases)", "",
             "| labels/option | head | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    groups = {}
    for r in result["runs"]:
        groups.setdefault((r["n"], r["rung"]), []).append(r)
    for (n, rung), rs in sorted(groups.items()):
        cells = []
        for c in cols:
            v = [r[c] for r in rs if r.get(c) is not None]
            cells.append("n/a" if not v else f"{np.mean(v):.3f}" + (f" ({np.std(v):.3f})" if len(v) > 1 else ""))
        label = "zero-data" if rung == 0 else f"rung {rung}"
        if rung == GUARDED:
            label = "train(), chose " + ",".join(str(r["chosen"]) for r in rs)
        n_txt = str(n) if n <= result["min_train_per_option"] else f"{n} (some fewer)"
        lines.append(f"| {n_txt} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", choices=suites.ALL)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="first N train and test cases only (smoke test)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from decision.backbone import DEFAULT_MODEL, GemmaBackbone

    dm = DecisionModel(GemmaBackbone(args.model or DEFAULT_MODEL))
    stamp = datetime.now()
    report = {"timestamp": stamp.isoformat(timespec="seconds"), "model": dm.backbone.name,
              "sizes": args.sizes, "seeds": args.seeds, "limit": args.limit, "suites": []}
    out = ROOT / "results" / f"learning-curves-{stamp:%Y%m%d-%H%M%S}"
    out.parent.mkdir(exist_ok=True)
    for name in args.suite or suites.ALL:
        report["suites"].append(run_suite(dm, name, args.sizes, args.seeds, args.limit))
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))  # after each suite
        md = "\n\n".join(summarize(r) for r in report["suites"])
        out.with_suffix(".md").write_text(f"# Learning curves ({report['timestamp']})\n\n{md}\n")
        print("\n" + summarize(report["suites"][-1]), flush=True)
    print(f"\nWrote {out.relative_to(ROOT)}.json and .md")


if __name__ == "__main__":
    main()
