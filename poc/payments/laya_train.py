"""Train Laya's multilingual encoder on the same generated examples the Gemma head learned from.

    python -m poc.payments.laya_train                 # writes poc/payments/laya/train.jsonl
    .vendor/laya/.venv/bin/laya-train --data poc/payments/laya/train.jsonl \\
        --base <multilingual checkpoint dir> --out poc/payments/laya/checkpoint --head-max-len 256
    .vendor/laya/.venv/bin/python -m poc.payments.bench laya --laya-checkpoint poc/payments/laya/checkpoint

Each training case becomes one row with the questions it labels and the expected answer to each,
in laya-train's JSONL format. A question whose phrase is still being typed in that prefix is left
out of the row, as it is left out of the Gemma head's training (labels.cases_from). States are
keyed by name, as Laya is asked at test time. Nothing here is committed: poc/payments/laya/ is
gitignored.
"""

import json
from pathlib import Path

from poc.search.jev import name_keys

from .filters import QUESTIONS
from .labels import OUT as CASES, leaks

HERE = Path(__file__).resolve().parent
OUT = HERE / "laya" / "train.jsonl"


def rows(cases):
    questions, back = name_keys(QUESTIONS)
    forward = {q: {v: k for k, v in m.items()} for q, m in back.items()}  # "IN" -> "indiana"
    for c in cases:
        keep = [k for k in QUESTIONS if k not in c.get("skip", [])]
        yield {"state": c["state"], "questions": {k: questions[k] for k in keep},
               "expected": {k: forward.get(k, {}).get(c["expected"][k], c["expected"][k]) for k in keep}}


def main():
    cases = json.loads(CASES.read_text())
    assert not leaks(cases), "training cases overlap the test set"
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        n = 0
        for row in rows(cases):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += len(row["questions"])
    print(f"{len(cases)} rows, {n} question labels -> {OUT}")


if __name__ == "__main__":
    main()
