# gemma4-decision-head

A classifier that reads decisions straight off a frozen Gemma 4, without generating any text.

Routing a message, scoring it on a scale and answering a yes/no question are all decisions with a fixed set of answers. Asking a language model to write the answer out is slow, can come back malformed, and gives you one answer with no idea how sure the model was. This project runs Gemma 4 for one forward pass and reads the answer from its final hidden state. Every option gets a probability, so you also get a confidence and a flag for when the model can't tell.

It works with no training data. When you have labelled examples, the same head trains on them in seconds, and 20 examples per option added 2.5 to 18.9 points of accuracy across 14 intent benchmarks.

It runs on Apple silicon with [MLX](https://github.com/ml-explore/mlx), on the 4-bit `mlx-community/gemma-4-e4b-it-4bit` checkpoint.

**[Write-up with learning curves](https://ai.gufamily.org/gemma4-decision-head)** · [Design doc](docs/DESIGN.md)

## Quick start

You need an Apple silicon Mac and Python 3.10 or later.

```bash
git clone https://github.com/Scooby-Dooby-Gu/gemma4-decision-head
cd gemma4-decision-head
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py
```

`evaluate.py` downloads the model on first use and scores the two sample suites in [`cases/`](cases/).

## Usage

You describe each decision as a question, then ask it about any piece of text.

```python
from decision import DecisionModel

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {
            "billing": "Payment, invoice, refund, or subscription issues",
            "technical": "Bugs, errors, outages, or integration problems",
            "sales": "Pricing, plan upgrades, or new account questions",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity",
    },
}

dm = DecisionModel()                                   # loads Gemma 4 once
answers = dm.answer("You charged my card TWICE!", questions)
```

There are three question types.

| Type | Use it for | `criteria` | The answer carries |
|---|---|---|---|
| `choice` | picking one of several named options | a dict of option name to description | `choice`, plus a probability for every option |
| `score` | rating on an ordered scale | a list of level descriptions, lowest first | `score` (the most likely level) and `expected` (the probability-weighted level) |
| `noul` | a yes/no statement | none | `noul`, the probability that the statement is true |

Every answer also has a `confidence`, which is the top probability, and an `abstain` flag, which is true when the top two options are within 0.10 of each other.

### Training on labelled examples

Labelled cases are a piece of text plus the expected answer to each question. Training starts from the zero-data head, so there is nothing to switch over, and a question with fewer than two labels per option stays zero-data.

```python
dm.fit(cases, questions)     # cases: [{"state": "...", "expected": {"department": "billing", ...}}]
dm.save_heads("heads/")
```

Or from the command line:

```bash
python train.py cases/treasury_ap_routing.json --out heads/
```

A trained head is tied to the exact wording of its question. If you edit the question, the model falls back to zero-data for it and never reuses a stale head.

### Serving over HTTP

```bash
python serve.py --heads heads/      # http://127.0.0.1:8092
curl -s localhost:8092/v1/systemone -d '{"state": "You charged my card TWICE!", "questions": {...}}'
```

The response is `{"model": ..., "answers": {...}}`, with the same answers `dm.answer` returns. `GET /health` reports that the model is loaded.

## How it decides

1. **Prompt.** The question becomes a prompt: the task, the options under short handles (letters for `choice`, digits for `score`, Yes/No for `noul`), the text to judge, then `Answer:`.
2. **One forward pass.** Gemma reads the prompt and stops. The head reads the final-layer hidden state `h` at the last position.
3. **Head.** `logit_k = softcap(U[handle_k] · h)`, where `U` is Gemma's own output embedding. With no training, this is exactly Gemma's next-token logit for each handle, restricted to the valid answers. A test checks that.
4. **Corrections.** Three corrections need no labels:
   - **Blank-input calibration.** Each question is scored once on a blank input (`N/A`), and 0.7 of that prior is divided out. This removes the model's lean towards particular answers.
   - **Order averaging.** `choice` questions are scored with the options in the listed order and in reverse, and the two are averaged. This cancels most position and letter bias.
   - **Clean handles.** A letter that appears as a word anywhere in the question, such as an option called "Candidate B", is not used as a handle.
5. **Output.** A softmax gives the probabilities.

`choice` questions take two forward passes because of the order averaging. The others take one.

### What training changes

| Rung | What is trained | What it fixes | Default for |
|---|---|---|---|
| 0 | nothing | | fewer than 2 labels per option |
| 1 | a bias per option and a temperature | over-picking one option, and confidence that doesn't match accuracy | `score`, `noul` |
| 2 | rung 1, plus a per-option correction read from the hidden state | options the descriptions alone can't separate | `choice` |

Training holds out a fifth of each option's labels as a quiz and fits every rung up to the default on the rest. If the default rung does worse on the quiz than a simpler one by more than 0.05 nats, the simpler rung is used. The quiz also sets the final temperature, because rung 2 tends to be overconfident. The chosen rung is then refitted on all the labels.

Rung 2 reads the hidden state scaled to unit length. Gemma 4's hidden states are large, and reading them directly made training oscillate, so that some runs ended far below the rest.

### Prefix caching

The part of the prompt before the text to judge is the same for every case, so its KV cache is computed once and reused. The cached path gives the same answers as full passes, with probabilities within 0.01 on average.

| Layout | What is cached | Good for |
|---|---|---|
| question first (default) | the question and its options, across cases | many cases per question |
| `state_first=True` (`--state-first`) | the text, across the questions asked about it | long texts with several questions |

On an M5 MacBook, caching the question cut the routing sample suite from 520 to 187 ms per case. Putting the text first made each case about 2 times faster at 300 tokens of text and 2.7 times faster at 1,300, but Gemma 4 E4B was a little less accurate with the question after the text (83% against 92% on one question). Check accuracy on your own cases before using that layout. A head trained in one layout is not reused in the other.

## Results

### Zero-data, on the sample suites

| Suite | Cases | Question | Accuracy |
|---|---|---|---|
| Support triage | 12 | department (`choice`) | 100% |
| | | is urgent (`noul`) | 91.7% |
| | | frustration (`score`) | 75.0% |
| Accounts-payable routing | 66 | agent (`choice`, 6 options) | 87.9% |

Both suites are small and hand-written. They show the head working with no labels, and they are too small to measure training.

### With labels, on public datasets

Accuracy on each dataset's test set, with no labels and after training on 20 labels per option. The trained figure is the mean of five random draws.

| Suite | Options | Zero-data | 20 labels per option | Gain |
|---|---|---|---|---|
| CLINC banking | 16 | 90.3% | 93.5% | +3.2 |
| CLINC credit cards | 16 | 93.8% | 96.3% | +2.5 |
| CLINC travel | 16 | 90.2% | 97.3% | +7.1 |
| Banking77, group 0 | 20 | 85.3% | 94.0% | +8.7 |
| Banking77, group 1 | 19 | 78.7% | 94.3% | +15.6 |
| Banking77, group 2 | 19 | 89.1% | 95.4% | +6.3 |
| Banking77, group 3 | 19 | 79.6% | 89.9% | +10.3 |
| **Held out** | | | | |
| CLINC kitchen and dining | 16 | 88.8% | 95.5% | +6.7 |
| CLINC home | 16 | 73.7% | 92.6% | +18.9 |
| CLINC auto and commute | 16 | 87.5% | 95.4% | +7.9 |
| CLINC utility | 16 | 81.8% | 96.3% | +14.5 |
| CLINC work | 16 | 87.5% | 96.3% | +8.8 |
| CLINC small talk | 16 | 88.3% | 94.1% | +5.8 |
| CLINC meta | 16 | 90.5% | 94.8% | +4.3 |

The training settings were chosen while looking at the first seven suites. The seven held-out CLINC domains were then run once with every setting frozen, and training also lowered calibration error on all seven.

On SST-5, a five-level sentiment rating, accuracy went from 47.8% to 53.1% and calibration error from 0.42 to 0.08. More labels did not help further there.

The option descriptions in these runs are just the intent names in words, so the zero-data figures are a floor. These datasets are also not your traffic. They show how the head trains, and you should measure it on your own cases.

Full tables, including 5 to 100 labels per option, are in [`results/`](results/).

## Evaluating your own cases

A suite is a JSON file with questions and labelled cases:

```json
{
  "name": "support_triage",
  "questions": {"department": {"type": "choice", "instructions": "...", "criteria": {"billing": "..."}}},
  "cases": [{"id": "t01", "state": "the text to judge", "expected": {"department": "billing"}}]
}
```

```bash
python evaluate.py --suite my_cases.json
python evaluate.py --folds 5       # also score trained heads with k-fold
python evaluate.py --no-debias     # listed order only, to measure order bias
python evaluate.py --no-cache      # a full pass per prompt, to time the prefix cache
python evaluate.py --state-first   # text before the question
```

It reports accuracy, calibration error (ECE), Brier score, abstain rate and handle mass, and writes the full results to `results/`.

**Handle mass** is the share of Gemma's own next-token probability that lands on the valid answer handles. The head only compares handles with each other, so it can look confident even when Gemma wanted to write something else. `evaluate.py` warns when the share is under 50%.

## Reproducing the learning curves

```bash
pip install -r requirements-experiments.txt
python -m experiments.learning_curves                        # all suites, about an hour the first time
python -m experiments.learning_curves --suite sst5 --sizes 0 10 50
python -m experiments.learning_curves --holdout              # the seven held-out CLINC domains
python -m experiments.learning_curves --limit 40             # a quick smoke test
```

The script caches Gemma's hidden states for every case once, which is the slow part. It then trains heads on 5, 10, 20, 50 and 100 labels per option, five random draws each, and scores them on the test set against the zero-data head.

The data comes from [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (one domain per suite, with its out-of-scope queries), [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (split into four groups, since 77 options are more than the head has handles for) and [SST-5](https://huggingface.co/datasets/SetFit/sst5).

## Project layout

```text
decision/
  model.py       DecisionModel: answer, fit, save and load heads.
  head.py        The head, its three rungs and the training loop.
  schema.py      Turns a question into prompts and answer handles.
  backbone.py    Loads Gemma 4, runs forward passes and keeps the prefix cache.
evaluate.py      Scores suites of labelled cases.
train.py         Trains heads from a suite and saves them.
serve.py         A small HTTP server.
experiments/     The learning-curve experiments on public datasets.
cases/           Two hand-written sample suites.
results/         Saved evaluation and learning-curve results.
docs/DESIGN.md   The original design doc.
tests/           Unit tests.
```

## Development

```bash
pip install pytest
python -m pytest tests -q
```

The tests run against a tiny, randomly initialised Gemma 4, so they need no download and finish in about 15 seconds. They check the mechanics: that the zero-data head equals Gemma's own logits, that a cached prefix gives the same hidden states as a full pass, that heads train, save and reload, and that an edited question drops its old head. They don't measure accuracy, which needs the real weights and `evaluate.py`.

## License

[MIT](LICENSE)
