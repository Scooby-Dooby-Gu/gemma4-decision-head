"""Upgrade path: train heads from labelled cases and save them for serve.py.

    python train.py ../jev-vs-local/cases/treasury_ap_routing.json --out heads/

The suite format is jev-vs-local's (questions + cases with `expected`). The rung is picked per
question from labels per option (under 5: stays zero-data; 5 to 49: bias and temperature;
50 and up: full probe). Force one with --rung.
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
    ap.add_argument("--rung", type=int, choices=[0, 1, 2])
    args = ap.parse_args()

    dm = DecisionModel(GemmaBackbone(args.model))
    for path in args.suite:
        suite = json.loads(Path(path).read_text())
        used = dm.fit(suite["cases"], suite["questions"], rung=args.rung)
        print(f"{suite['name']}: " + ", ".join(f"{k} rung {r}" for k, r in used.items()))
    dm.save_heads(args.out)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
