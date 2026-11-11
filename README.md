# Gemma 4 decision model (no generation)

A classifier on top of the frozen `mlx-community/gemma-4-e4b-it-4bit` backbone. It takes a
state plus Jev-style questions (`choice`, `score`, `noul`) and returns a decision with a full
probability distribution, a confidence and an abstain flag. It never decodes a token: each
question is one forward pass (two for `choice`), and the answer is read off the final hidden
state. It works with zero training data, and labelled cases upgrade the same head in place.

Design doc: [docs/DESIGN.md](docs/DESIGN.md)

## How it decides

1. The question becomes a prompt: task, options under short handles (letters for `choice`,
   level digits for `score`, Yes/No for `noul`), then the state, then `Answer:`. For `score`
   it is `Answer: ` with a trailing space: Gemma writes " 1" as a space then a digit, so
   without the space it puts almost no probability on the digits.
2. Gemma runs one forward pass. The final-layer hidden state `h` at the last position (after
   the final norm) is the pooled representation the head reads.
3. Head: `logit_k = softcap(U[handle_k] . h)`, where `U` is Gemma's own output embedding. With
   no data this is exactly Gemma's next-token logit for that handle (a test checks this),
   restricted to the valid answers.
4. Three label-free corrections:
   - **Content-free calibration.** Each question is scored once on a blank state (`N/A`), and
     0.7 of that log-prior is divided out.
   - **Order debiasing.** `choice` questions are scored in the listed and the reversed order,
     and the per-option log-probabilities are averaged.
   - **Clean handles.** Letters that appear as a standalone token anywhere in the question
     (an option called "Candidate B", a sentence starting "A") are skipped.
5. Softmax gives the probabilities. `confidence` is the top probability, and `abstain` is
   true when the margin to the runner-up is under 0.10.

## Upgrade path (labelled data)

Labelled cases use the jev-vs-local format (`state` + `expected`). The head that gets trained
is the zero-data head, so zero-data is its starting point and there is nothing to switch over.
The rung is picked per question from labels per option:

| Labels per option | Trained | Fixes |
|---|---|---|
| under 5 | nothing (zero-data) | |
| 5 to 49 | per-option bias + temperature | over-picking an option; confidence vs. accuracy |
| 50 and up | also a per-option correction to `U[handle_k]`, decayed toward zero | options the descriptions alone can't separate |

Heads are keyed by a fingerprint of the question, so editing a question falls back to
zero-data rather than reusing a stale head.

## Install

Needs Apple silicon (MLX) and Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

`evaluate.py` scores every suite in [`cases/`](cases/) by default (see [Data](#data)); pass
`--suite` to point at others.

```bash
# Zero-data accuracy, ECE, Brier, abstain rate and handle mass on every suite
python evaluate.py
python evaluate.py --folds 5     # plus k-fold scores for the upgrade head
python evaluate.py --no-debias   # listed order only, to measure order bias

# Train heads from labelled cases, then serve Jev's /v1/systemone on :8092
python train.py cases/treasury_ap_routing.json --out heads/
python serve.py --heads heads/
```

From Python:

```python
from decision import DecisionModel
dm = DecisionModel()                      # loads Gemma 4 once
dm.answer("You charged my card TWICE!", questions)
dm.fit(cases, questions)                  # optional
dm.save_heads("heads/")
```

Handle mass is the share of Gemma's own next-token probability that lands on the valid answer
handles. The head only compares handles with each other, so it looks confident even when Gemma
wants to write something else; `evaluate.py` warns when the share is under 50%.

Results go to `results/`. `results/eval-20261111-124057.json` is a zero-data run with
`--folds 5` on the real 4-bit E4B weights (support triage, 12 cases: department 100%, urgency
91.7%, frustration 75.0%; treasury AP routing, 66 cases: 87.9%, 90.9% with a trained head).
The earlier `results/eval-20261018-200307.json` predates the score-prefix fix (frustration
66.7%, with handle mass 0.003).

### Data

A suite is a JSON file `{"name", "questions": {key: question}, "cases": [{"id", "state",
"expected": {key: answer}}]}`, where a question is `choice` (criteria dict), `score` (list of
level descriptions) or `noul` (a yes/no statement). Two hand-written sample suites are included:

| Suite | Cases | Questions |
|---|---|---|
| `cases/support_triage.json` | 12 | `department` (choice), `frustration` (score 0 to 2), `is_urgent` (noul) |
| `cases/treasury_ap_routing.json` | 66 | `agent` (choice of five agents or out of scope), with tags such as `clear` and `terse` |

They are small and synthetic: enough to show the zero-data result, too few to both train and
measure a probe.

## Tests

`tests/test_decision.py` runs against a tiny, randomly initialised, 4-bit quantized Gemma 4
built from mlx-vlm's own Gemma 4 code, so it needs no download and finishes in about 15 s:

```bash
pip install pytest
python -m pytest tests -q
```

It checks that the zero-data head equals Gemma's own logits, that score prompts end on the
space, that handle mass is computed right, that answers have Jev's shape,
that calibration and order averaging behave, that the rungs train, save and reload, and that
an edited question or prompt template drops its old head. It does not measure accuracy: that needs the real
weights, which `evaluate.py` uses.

## Not done yet

- **Prefix caching.** The schema part of the prompt is recomputed for every case. Caching
  its KV state would cut latency, but Gemma 4's sliding-window cache makes that fiddly, so it
  waits for real latency numbers.
- **jev-vs-local backend.** `serve.py` speaks the same protocol as Von, Laya and Winnow, so a
  `gemma_head` backend there is a four-line subclass of `LocalSystemOneBackend`. It isn't
  added because it changes existing jev-vs-local files.
