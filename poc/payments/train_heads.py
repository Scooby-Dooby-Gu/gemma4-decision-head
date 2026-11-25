"""Train the payment filter heads on the generated cases in train_cases.json (see labels.py).

    python -m poc.payments.labels          # (re)generate train_cases.json
    python -m poc.payments.train_heads     # -> poc/payments/heads/

The heads are trained in the phase 0 setup (listed option order, no blank-input prior), so load
them only into a model set up the same way: `python -m poc.payments.bench head --heads poc/payments/heads`. Hidden states are cached in
poc/payments/cache/, so retraining with other settings skips the backbone passes.
"""

import json
import time
from pathlib import Path

import numpy as np

from .filters import QUESTIONS
from .labels import OUT as CASES, leaks

HERE = Path(__file__).resolve().parent
HEADS = HERE / "heads"
CACHE = HERE / "cache" / "train-features.npz"


def main():
    cases = json.loads(CASES.read_text())
    assert not leaks(cases), "training cases overlap the test set"
    from decision import DecisionModel

    dm = DecisionModel(debias_order=False)
    dm.answer("warm up", QUESTIONS)
    for s in dm.schemas.values():  # no blank-input prior, as in bench.head_engine
        for view in s.views:
            view.prior = None
    keys = list(QUESTIONS)
    texts = "\n".join(c["state"] for c in cases)
    if CACHE.exists() and str(np.load(CACHE)["texts"]) == texts:
        d = np.load(CACHE)
        feats = {k: ([d[f"{k}_H"]], d[f"{k}_y"]) for k in keys}
        print(f"hidden states from {CACHE}")
    else:
        t = time.perf_counter()
        feats = dm.features_all(cases, QUESTIONS)
        print(f"hidden states for {len(cases)} cases in {time.perf_counter() - t:.0f} s")
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, texts=texts, **{f"{k}_H": feats[k][0][0] for k in keys},
                 **{f"{k}_y": feats[k][1] for k in keys})
    t = time.perf_counter()
    used = {}
    for key, q in QUESTIONS.items():
        # Each question trains on the cases that label it: not on prefixes that stop partway
        # through the phrase for this question (see labels.cases_from).
        keep = np.array([key not in c.get("skip", []) for c in cases])
        H, y = feats[key]
        used.update(dm.fit([c for c, k in zip(cases, keep) if k], {key: q},
                           cached={key: ([h[keep] for h in H], y[keep])}))
    print(f"trained in {time.perf_counter() - t:.0f} s")
    for key, q in QUESTIONS.items():
        head = dm.heads[(key, dm.schema(key, q).fingerprint)]
        sel = head.selection or {}
        quiz = {r: round(v, 3) for r, v in (sel.get("quiz_loss") or {}).items()}
        print(f"  {key}: rung {used[key]} on {head.n_labels} labels (guarded {sel.get('guarded')}, quiz loss {quiz})")
    dm.save_heads(HEADS)
    print(f"saved heads to {HEADS}")


if __name__ == "__main__":
    main()
