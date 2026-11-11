"""Score the decision model on jev-vs-local suites: accuracy, calibration and latency.

    python evaluate.py                                   # zero-data, every suite
    python evaluate.py --suite cases/support_triage.json
    python evaluate.py --folds 5                         # also k-fold the upgrade head
    python evaluate.py --no-debias                       # listed order only (fewer passes)
    python evaluate.py --state-first                     # state before question (for long states)
    python evaluate.py --no-cache                        # full pass per view, to time the cache

Hidden states are computed once per case and cached, so --folds only adds seconds.
"""

import argparse
import glob
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from decision import DecisionModel
from decision.head import train as train_head

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases"
MASS_CASES = 5    # cases per question checked for handle mass
LOW_MASS = 0.5


def metrics(p, y, abstain_margin):
    """p (N, K) probabilities, y (N,) labels."""
    top = p.argmax(1)
    conf = p.max(1)
    correct = top == y
    onehot = np.eye(p.shape[1])[y]
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    srt = np.sort(p, 1)
    keep = (srt[:, -1] - srt[:, -2]) >= abstain_margin
    return {
        "accuracy": float(correct.mean()),
        "ece": float(ece),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "abstain_rate": float(1 - keep.mean()),
        "accuracy_when_answering": float(correct[keep].mean()) if keep.any() else None,
    }


def kfold(dm, s, H, y, folds, seed=0):
    """Out-of-fold probabilities from heads trained on the other folds."""
    idx = np.random.default_rng(seed).permutation(len(y))
    out, rungs = np.zeros((len(y), len(s.options))), []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        head = train_head(s, [h[train] for h in H], y[train], H[0].shape[-1], dm.backbone.softcap, seed=seed)
        rungs.append(head.rung)
        out[test] = head.predict(s, [h[test] for h in H])
    return out, rungs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append")
    ap.add_argument("--model", default=None, help="MLX checkpoint (default gemma-4-e4b-it-4bit)")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--no-debias", action="store_true")
    ap.add_argument("--state-first", action="store_true", help="state before the question (cache the state)")
    ap.add_argument("--no-cache", action="store_true", help="no prefix KV cache: a full pass per view")
    args = ap.parse_args()

    from decision.backbone import DEFAULT_MODEL, GemmaBackbone

    dm = DecisionModel(GemmaBackbone(args.model or DEFAULT_MODEL), debias_order=not args.no_debias,
                       state_first=args.state_first, use_cache=not args.no_cache)
    cached = dm.use_cache
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": dm.backbone.name,
              "debias_order": dm.debias_order, "state_first": dm.state_first, "prefix_cache": cached,
              "suites": {}}
    for path in args.suite or sorted(glob.glob(str(DEFAULT_CASES / "*.json"))):
        suite = json.loads(Path(path).read_text())
        cases = suite["cases"]
        print(f"\n== {suite['name']} ({len(cases)} cases)")
        schemas = {key: dm.schema(key, q) for key, q in suite["questions"].items()}
        # All questions about a case are answered together, as serve.py does, so the state is
        # read once per case; time is per case for every question.
        t0 = time.perf_counter()
        feats = dm.features_all(cases, suite["questions"])
        ms = (time.perf_counter() - t0) * 1000 / len(cases)
        n_views = sum(len(s.views) for s in schemas.values())
        print(f"  {ms:.0f} ms/case for all {len(schemas)} questions ({n_views} views,"
              f" {('cached ' + ('state' if dm.state_first else 'question') + ' prefix') if cached else 'full pass per view'})")
        report["suites"][suite["name"]] = res = {"ms_per_case": ms}
        for key, s in schemas.items():
            H, y = feats[key]
            p = dm.heads[(key, s.fingerprint)].predict(s, H)
            m = metrics(p, y, dm.abstain_margin)
            # Share of Gemma's probability on the answer handles (see DecisionModel.handle_mass),
            # on the first few cases: a low share means the prompt is not set up to be answered.
            m["handle_mass"] = float(np.mean([min(dm.handle_mass(s, c["state"])) for c in cases[:MASS_CASES]]))
            res[key] = {"zero_data": m, "predictions": [
                {"id": c["id"], "expected": c["expected"][key], "got": s.options[int(pi.argmax())],
                 "confidence": float(pi.max())} for c, pi in zip(cases, p)]}
            print(f"  {key:<14} zero-data  acc {m['accuracy']:.1%}  ECE {m['ece']:.3f}  Brier {m['brier']:.3f}"
                  f"  abstain {m['abstain_rate']:.0%}  handle mass {m['handle_mass']:.2f}")
            if m["handle_mass"] < LOW_MASS:
                print(f"  WARNING: {key}: Gemma puts only {m['handle_mass']:.0%} of its probability on the"
                      " answer handles, so these probabilities are not meaningful. Check the prompt.")
            if args.folds:
                pf, rungs = kfold(dm, s, H, y, args.folds)
                mf = metrics(pf, y, dm.abstain_margin)
                mf["rungs"] = rungs
                res[key]["kfold"] = mf
                print(f"  {'':<14} {args.folds}-fold head acc {mf['accuracy']:.1%}  ECE {mf['ece']:.3f}"
                      f"  Brier {mf['brier']:.3f}  rungs {sorted(set(rungs))}")
    out = HERE / "results" / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
