"""Upgrade path: train heads from labelled cases and save them for serve.py.

    python train.py cases/treasury_ap_routing.json --out heads/

The suite format is jev-vs-local's (questions + cases with `expected`). Each question gets the
rung for its type (2 for choice, 1 for score and noul), checked against simpler rungs and
calibrated on a held-out fifth of its labels, then refitted on all of them (see decision/head.py).
--rung sets the highest rung tried. Questions with under two labels per option stay zero-data.
"""

import argparse
import json
from pathlib import Path

from decision import DecisionModel
from decision.backbone import DEFAULT_MODEL, GemmaBackbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="+")
    ap.add_argument("--out", default="heads")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--state-first", action="store_true",
                    help="state before the question; must match how the heads were trained")
    ap.add_argument("--rung", type=int, choices=[0, 1, 2])
    args = ap.parse_args()

    dm = DecisionModel(GemmaBackbone(args.model), state_first=args.state_first)
    for path in args.suite:
        suite = json.loads(Path(path).read_text())
        used = dm.fit(suite["cases"], suite["questions"], rung=args.rung)
        print(f"{suite['name']}:")
        for key, q in suite["questions"].items():
            sel = dm.heads[(key, dm.schema(key, q).fingerprint)].selection
            if sel is None:
                print(f"  {key}: zero-data (too few labels to hold out a quiz)")
                continue
            quiz = ", ".join(f"rung {r} {v:.3f}" for r, v in sel["quiz_loss"].items())
            fell = f", fell back from rung {sel['requested']}" if sel["chosen"] < sel["requested"] else ""
            if not sel["guarded"]:
                fell += ", guard off: under 2 quiz labels per option"
            print(f"  {key}: rung {used[key]}{fell} (quiz of {sel['quiz_size']}, loss {quiz})")
    dm.save_heads(args.out)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
