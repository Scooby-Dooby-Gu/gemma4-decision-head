# Gemma 4 decision model (no generation)

A classifier on top of the frozen `mlx-community/gemma-4-e4b-it-4bit` backbone. It takes a
state plus Jev-style questions (`choice`, `score`, `noul`) and returns a decision with a full
probability distribution, a confidence and an abstain flag. It never decodes a token: each
question is one forward pass (two for `choice`) over the state and a short suffix, with the
question's own prefix read once and cached, and the answer is read off the final hidden state. It works with zero training data, and labelled cases upgrade the same head in place.

Design doc: [docs/DESIGN.md](docs/DESIGN.md)

## How it decides

1. The question becomes a prompt: task, options under short handles (letters for `choice`,
   level digits for `score`, Yes/No for `noul`), then the state, then `Answer:`. For `score`
   it is `Answer: ` with a trailing space: Gemma writes " 1" as a space then a digit, so
   without the space it puts almost no probability on the digits.
2. Gemma runs one forward pass. The final-layer hidden state `h` at the last position (after
   the final norm) is the pooled representation the head reads. The part of the prompt before
   the state is the same for every case, so its KV cache is computed once and reused (see
   [Prefix caching](#prefix-caching)).
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

| Rung | Trained | Fixes | Default for |
|---|---|---|---|
| 0 | nothing (zero-data) | | under 2 labels per option |
| 1 | per-option bias + temperature | over-picking an option; confidence vs. accuracy | `score`, `noul` |
| 2 | also a per-option correction read from the unit-length hidden state | options the descriptions alone can't separate | `choice` |

The defaults come from the learning curves below: rung 2 beat rung 1 on every intent suite from
5 labels per option, and hurt on SST-5's 5-level ratings. `train` (used by `DecisionModel.fit`
and `train.py`) holds out a fifth of each option's labels as a quiz and fits every rung up to
the default on the rest. The quiz is a guard: if the default rung's loss there trails a simpler
rung by more than 0.05 nats, the simpler one is used. The guard needs two quiz labels per option
(about 10 labels per option); with fewer it misfired, so the default rung is kept. It also sets the final temperature, since
rung 2 tends to be overconfident. The chosen rung is then refitted on all labels.

Rung 2's correction reads `h / |h|`, not `h`. Gemma 4's hidden states are large (`|h|_1` is
about 5,000), and reading them directly let one optimizer step move a score by ~25, so training
oscillated, and in 7 of 32 learning-curve settings a run ended mid-spike (as low as 54% where
the other draws reached 89%). Heads saved before this change are refused at load if they are rung 2.

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
python evaluate.py --no-cache    # full pass per view, to time the prefix cache
python evaluate.py --state-first # state before the question (see Prefix caching)

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

### Prefix caching

Each prompt is split at the state into a shared part, whose KV cache is computed once and kept
(the 64 most recent prefixes), and the rest, which runs per call. The cached path gives the
same answers as full passes, and probabilities within 0.01 on average (at most 0.085 on one
case), which is 4-bit floating-point drift.

| Layout | Shared and cached | Good for |
|---|---|---|
| question first (default) | the question and its options, across cases | many cases per question; the prompt is unchanged |
| `state_first=True` (`--state-first`) | the state, across the questions about one case | long states with several questions |

On the sample suites, the default cuts treasury AP routing from 520 to 187 ms per case and
support triage (short states, three questions) from 550 to 536 ms. Reading the state first
made each case about 2x faster at ~300 state tokens and 2.7x at ~1,300, but Gemma 4 E4B is a
little less accurate with the question after the input (is_urgent 83% vs 92%), and the
blank-state calibration over-corrects there, so it is off for that layout. Heads trained in
one layout are not reused in the other.

Handle mass is the share of Gemma's own next-token probability that lands on the valid answer
handles. The head only compares handles with each other, so it looks confident even when Gemma
wants to write something else; `evaluate.py` warns when the share is under 50%.

Results go to `results/`. `results/eval-20261111-125800.json` is the same run with the prefix
cache (same accuracy). `results/eval-20261111-124057.json` is a zero-data run with
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

## Experiments: how many labels does training need?

`experiments/learning_curves.py` measures the upgrade path on public datasets, which have
enough labels to train on and a separate test set to score on:

| Suite | Type | Options | Source |
|---|---|---|---|
| `clinc_banking`, `clinc_credit_cards`, `clinc_travel` | choice | 15 intents + `out_of_scope` | [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), one domain each, with its labelled out-of-scope queries |
| `banking77_g0` to `banking77_g3` | choice | 19 to 20 intents | [Banking77](https://huggingface.co/datasets/PolyAI/banking77), split in four (77 options exceed the ~34 handles) |
| `sst5` | score | 5 levels | [SST-5](https://huggingface.co/datasets/SetFit/sst5) movie-review sentiment |

For each suite it caches Gemma's hidden states for every case once (the slow part, about
0.2 s per case), then trains heads on 5, 10, 20, 50 and 100 labels per option, five random
draws each, at rung 1 and rung 2, and scores them on the test set against the zero-data head.
CLINC suites also report out-of-scope recall and precision.

```bash
pip install -r requirements-experiments.txt
python -m experiments.learning_curves                      # all suites, ~1 h the first time
python -m experiments.learning_curves --suite sst5 --sizes 0 10 50
python -m experiments.learning_curves --limit 40           # smoke test
```

Results go to `results/learning-curves-<time>.json` and a `.md` summary. Option descriptions
are the intent names in words, so zero-data accuracy is a floor. These datasets are not your
traffic: they show how the head trains, not how it will do on your cases.

## Tests

`tests/test_decision.py` runs against a tiny, randomly initialised, 4-bit quantized Gemma 4
built from mlx-vlm's own Gemma 4 code, so it needs no download and finishes in about 15 s:

```bash
pip install pytest
python -m pytest tests -q
```

It checks that the zero-data head equals Gemma's own logits, that score prompts end on the
space, that handle mass is computed right, that answers have Jev's shape, that calibration and
order averaging behave, that a cached prefix gives the same hidden states as full passes in
both layouts (including past the sliding window), that the rungs train, save and reload, and
that an edited question or prompt template drops its old head. It does not measure accuracy:
that needs the real weights, which `evaluate.py` uses.

## Not done yet

- **jev-vs-local backend.** `serve.py` speaks the same protocol as Von, Laya and Winnow, so a
  `gemma_head` backend there is a four-line subclass of `LocalSystemOneBackend`. It isn't
  added because it changes existing jev-vs-local files.
