# Zero-data decision head on Gemma 4

Oct 16, 2026 · Simon Gu

_Exported from the original design doc. Some numbers here (Gemma accuracy and latency) were estimates written before the head was built; measured results are in `results/`._

## Summary

Run Gemma 4 E4B for **one forward pass per question, never a decode loop**, and read the decision straight off the hidden state at the answer position. With no data, a linear head built from Gemma's own output embeddings for the option tokens ("A", "B", "Yes", "No", "0", "1"…) gives a full probability distribution over the options. When labelled cases arrive, that same head is trained, starting from those weights, so the zero-data model is literally the initialisation of the trained one and there is no switch-over.

The model speaks the same Jev `/v1/systemone` format that jev-vs-local already scores, so it slots in as a new backend next to `gemma`, `winnow`, `von` and `laya`. Unlike today's `gemma` backend it returns real probabilities and a confidence, which is the README's listed next step.

## Pros and cons

The approach is expected (not yet measured) to match today's generative Gemma on accuracy while gaining speed, real probabilities and a smooth upgrade path. Its weaknesses are a ceiling set by the backbone and the biases it inherits from it.

**Pros**

- **Works on day one.** No training data, and any schema defined at request time is answerable immediately.
- **Real probabilities.** Every option gets a probability, so callers get a confidence and an abstain signal. Today's `gemma` backend returns a single answer with confidence `None`.
- **No decode loop.** One forward pass replaces writing JSON. On the same kind of 4B model, SemIf measured 1.0 s for 21 answers read from logits versus 5.3 s to generate them as a JSON array (RTX 3090). The Gemma figure on the M5 is not measured yet.
- **Cannot return a malformed answer.** The output is always one of the declared options, so there is no JSON parsing or repair.
- **Continuous upgrade path.** The zero-data head is the starting point of the trained head, so labels improve it without a switch-over, and the early rungs train in seconds on the laptop.
- **Reuses what exists.** Same checkpoint, same MLX runtime, same Jev request format, so it drops into jev-vs-local as one more backend.

**Cons**

Each con below says why it happens, how much it matters for this project, and what limits it.

- **Ceiling is the backbone's judgment.**
  - Why: the head only reads what Gemma already thinks at the answer position. A linear layer can re-weight that view but cannot add facts or understanding Gemma lacks.
  - How much: zero-data accuracy should land near generative Gemma, 93.9% on treasury, which leaves about 4.6 points to Jev's 98.5%. On 66 cases that is three cases.
  - What limits it: the probe rung recovers some errors where Gemma's hidden state holds the right answer but its letter token does not. Closing the rest needs LoRA, a larger backbone (SemIf's 27B scored 0.958 against 0.813 for the 4B on its authored set), or routing low-confidence cases to Jev.
- **Inherits the backbone's biases.**
  - Why: the scores are Gemma's own next-token preferences, which carry habits from pretraining and instruction tuning, such as favouring certain positions or Yes.
  - How much: on Qwen3.5-4B with the same readout, reversing the options changed 10 of 36 answers and rewording the question changed 9. Gemma has not been measured.
  - What limits it: order averaging, clash-free answer letters and content-free calibration (see Inherent biases). Order averaging doubles the passes, which eats part of the speed gain. Rewording has no runtime fix.
- **No room to reason.**
  - Why: one forward pass gives the model no tokens to work through a problem. Generation with a reasoning step can compare options, do arithmetic or apply a rule in stages before answering.
  - How much: small for routing and triage, where the answer is mostly recognition. Larger for questions such as "is this payment above the approver's limit" that need a calculation or several rules combined. Today's `gemma` backend does not reason either, so nothing is lost against it.
  - What limits it: let Gemma reason first and then read the option logits after its reasoning. That restores the reasoning at the cost of the decode loop, so it belongs only on the questions that need it.
- **Uncalibrated until labels exist.**
  - Why: instruction-tuned models are trained to answer decisively, so the probability on the top option is often higher than its real hit rate.
  - How much: SemIf's Qwen was right 64% of the time on WANLI while reporting much higher confidence; a temperature of 2.5 was needed. Without labels, a threshold such as "act above 0.9" has no guaranteed meaning.
  - What limits it: rank by margin rather than trusting the absolute number at first, offer an `insufficient` option, and fit a temperature from the first 20 or so labelled cases per workload.
- **Trained heads are per schema.**
  - Why: a probe learns one weight row per option, so it only knows the options it was trained on. Renaming an option or adding one changes the head's shape.
  - How much: high if question sets change often, as they can with Jev-style runtime schemas. Low for stable workloads such as the treasury router.
  - What limits it: a changed schema falls back cleanly to zero-data mode rather than breaking. A head that works across schemas, scoring each option from its own hidden state as Laya does, removes the limit but needs many labelled schemas to train.
- **Needs model internals.**
  - Why: the readout uses logits or hidden states that a chat endpoint such as `mlx_vlm.server` does not expose.
  - How much: one-off engineering, a small Python server that loads the model in MLX and serves `/v1/systemone`. Each answer handle must be a single token, which caps the options per question at the size of the handle set (SemIf allows 16).
  - What limits it: SemIf already has an MLX backend that does this for Qwen and could be adapted to Gemma.
- **Heavier than the small local models.**
  - Why: E4B is a 4-billion-effective-parameter decoder; Von and Laya are 395M to 421M-parameter encoders.
  - How much: about 5.3 GB of memory against 1.5 GB for Von, on a 24 GB laptop. Von and Laya answer in 64 to 104 ms; Gemma's single-pass latency on the M5 is not yet measured but will be higher.
  - What limits it: prefix caching means each case only processes its own state. The accuracy gap (Von 59.1% on treasury) currently outweighs the speed gap.

**Against the alternatives**

| Approach | Main pro | Main con |
| --- | --- | --- |
| This design: option logits plus a trainable head | Probabilities with zero data, and one path from zero to labelled | Backbone ceiling and inherited biases |
| Generative JSON (today's `gemma` backend) | Already works, 93.9% on treasury | No probabilities, slower, can emit malformed output |
| Embedding similarity to option descriptions | Cheapest per option, scales to many options | Decoder states make weak embeddings; likely less accurate |
| Decision fine-tune (Winnow-E4B) | Built for decisions | Needs a large labelled corpus up front; scored below plain Gemma on treasury (90.9%) |
| Small encoder (Von, Laya) | Fastest locally, 64 to 104 ms | Much lower accuracy here (51.5% to 75%) |
| Hosted Jev | Most accurate, 98.5% on treasury | Not local, paid per call, closed model |

## Architecture

The backbone is the existing `mlx-community/gemma-4-e4b-it-4bit` checkpoint, frozen and loaded once in MLX. Only the head is new, and it is a few kilobytes per question.

1. **Template.** Each question becomes one input: the question's instructions and its options lettered A, B, C… first, then the case's `state`, then a fixed suffix ending in `Answer:`. The schema part comes first so its key-value cache is computed once per suite and reused for every case.
2. **Forward pass.** Gemma runs over the state and suffix only (the schema prefix is cached). No tokens are sampled.
3. **Read.** Take the final-layer hidden state `h` at the last position, after Gemma's final norm. A later rung can read a middle layer instead.
4. **Head.** `logits = W h + b`, one row of `W` per option. For a `choice` question the rows are the options' letter tokens, for `score` the level digits, for `noul` the tokens Yes and No.
5. **Calibrate and answer.** Subtract the content-free bias (next section), apply a temperature, softmax, and return Jev-shaped `answers` with `probabilities`, `confidence`, and an abstain flag when the margin is too small.

```
state ─► prompt (options + state + "Answer:") ─► frozen Gemma 4, one pass ─► h (last hidden state)
h ─► softcap(U[handle_k] · h) ─► minus 0.7 × blank-state prior ─► + b, ÷ T ─► softmax
  ─► choice / score / noul, probabilities, confidence, abstain
```

Only the head changes as data arrives; the prompt, backbone and calibration step stay the same.

Options are labelled with letters rather than the criteria keys, because multi-token keys such as `payment_search` or `balance_liquidity` can share a first token with other keys, and a single forward pass only scores one token. The letter is only a handle: the model still reads each option's full criteria description in the prompt.

## Zero-data mode

With no labels, the head is Gemma's own unembedding: each row of `W` is the output-embedding row for that option's token, and `b` is zero. This is exactly the probability Gemma would put on writing "B" next, restricted to the valid answers, so it should land close to the accuracy today's `gemma` backend shows (93.9% on treasury, 80.6% on support triage in the 27 September run) while costing one prefill instead of a JSON decode.

| Question type | Tokens scored | Answer returned |
| --- | --- | --- |
| `choice` | one letter per option | argmax option, full distribution |
| `score` | one digit per level | argmax level; expected value kept for ordinal use |
| `noul` | Yes, No | P(Yes) / (P(Yes) + P(No)) |

Three corrections need no labels at all:

- **Content-free calibration.** Run each schema once with a blank state ("N/A") and divide out the resulting prior, which removes Gemma's preference for particular letters or for Yes.
- **Order debiasing.** Score each case in the listed and reversed option order and average. It costs one extra pass, and the evidence in the next section says it is worth it.
- **Abstain.** Report the top-1 probability as `confidence` and flag answers whose margin over the runner-up is below a threshold, so a caller can route them to a human or to Jev.

Embedding similarity (cosine between a pooled state vector and each option's description vector) was considered as the zero-data route and rejected as the primary. Decoder hidden states are not trained as sentence embeddings and tend to cluster, so it would likely score below the verbalizer. It stays useful later as a cheap shortlist for schemas with dozens of options.

## Inherent biases

Reading option logits inherits every preference the backbone has, and the closest public evidence says these are large enough to change answers. [SemIf](https://github.com/TheoLeeCJ/openjev) (formerly OpenJev) uses this same readout, letter-slot logits from one forward pass, on Qwen3.5-4B and applies no debiasing, only a temperature. Its committed results show what an uncorrected head does:

| Bias | Evidence on Qwen3.5-4B (SemIf, commit 23cf1f3) | Fix in this design | Extra cost |
| --- | --- | --- | --- |
| Option order | Reversing the options flipped 10 of 36 base decisions. Only 1 of the 10 flips kept the same letter, so it is not a fixed "prefers A" habit a single prior could remove. | Score in the listed order and reversed, average the log-probabilities per option, on by default for `choice` | one extra state pass; each order's schema prefix is cached |
| Handle collision | Where options are named "Candidate A" and "Candidate B", the order that put Candidate B under letter A picked the wrong candidate in 8 of 15 rows. The other orders had none of those swaps. | Pick handles that appear nowhere in the option ids or descriptions (for example X, Y, Z, or digits for `choice`) | none |
| Answer prior (Yes-bias, favourite options) | Not isolated by SemIf. Von, a different model in `vendor/`, measures a Yes preference with a blank state and found subtracting it fully over-corrects, so it subtracts 0.7 of it. | Content-free calibration with a shrink factor, starting at 0.7 and fitted once labels exist | one pass per schema, cached |
| Overconfidence | On WANLI the model was right 64% of the time while far more confident; the fitted temperature was 2.5 and expected calibration error fell from 0.208 to 0.069. It also picked a definite answer above 0.8 on a case with missing evidence. | Temperature per workload once labels exist; always offer an `insufficient` option where the schema allows it; abstain on low margin | none |
| Wording | Rewording the criterion flipped 9 of 36; irrelevant extra context flipped 4. | Freeze one template; flag schemas whose answers move under a rewording in the eval | none at runtime |

One kind of bias is missing from the table because SemIf did not measure it: social and demographic bias learned in pretraining. None of the fixes above touch it, since they correct the answer distribution, not what the model associates with a name or a group. If this model ever decides anything about people, it needs a counterfactual test first: the same case with names, genders or locations swapped, measuring how often the decision moves.

SemIf's authored and perturbation fixtures (144 and 108 rows, project-authored, MIT) can run against the Gemma head unchanged, which gives a before-and-after for each fix on Gemma itself rather than relying on Qwen numbers.

## Upgrade path

Labelled cases use the format jev-vs-local already has (`state` plus `expected`), so any suite in `cases/` is also training data. Each rung trains more of the head, always starting from the rung below it, and falls back to it when a schema has too few labels.

| Labels per option | What is trained | Trainable size | What it fixes |
| --- | --- | --- | --- |
| 0 | nothing | 0 | baseline (zero-data mode) |
| about 5 to 20 | bias `b` and temperature | one number per option, plus one | systematic over-picking of an option; confidence that matches accuracy |
| about 50 to 500 | full `W` and `b`, initialised from the unembedding rows, with weight decay pulling back toward them | options × hidden size (a few thousand to tens of thousands of weights) | options Gemma confuses that the descriptions alone cannot separate |
| thousands, across many schemas | a small scoring head over every option's hidden state, or LoRA adapters on the backbone | roughly 1 to 50 million parameters | the gap to a decision fine-tune such as Winnow-E4B or Jev |

The first two rungs train in seconds on the laptop from logits cached during an eval run. The probe rung needs one cached hidden vector per case and also trains in seconds. A layer sweep is worth doing here, since middle layers often probe better than the last.

The probe rungs are per schema: a new set of options starts back at zero-data mode. The last rung is the one that generalises across schemas. It follows Laya's `DecisionModel` (in `vendor/laya-src`), which scores a hidden state taken at a marker after each option; with Gemma being causal, the options go after the state so each marker sees it. That rung needs many labelled schemas and is the most expensive, so it waits until the earlier rungs show where they stall.

## How it fits and how to judge it

Build it as a small local server answering `/v1/systemone`, plus a `gemma_head` backend in `jev-vs-local/backends/`, so `run.py` compares it with no other changes. The bar it has to clear, from the latest runs in `jev-vs-local/results/`:

| Backend | Treasury AP routing (66 cases) | Support triage (12 cases) | Median latency |
| --- | --- | --- | --- |
| Jev 1.13 (hosted) | 98.5% | not run | 221 ms |
| Gemma 4 E4B, generative | 93.9% | 80.6% | 438 to 753 ms |
| Winnow-E4B | 90.9% | 86.1% | 391 to 410 ms |
| Von | 59.1% | 75.0% | 64 to 104 ms |
| Laya | 51.5% | 69.4% | 64 to 71 ms |

Accuracy alone is not enough to judge the change, since the zero-data head should roughly match generative Gemma by design. The wins to measure are latency without the decode, and calibration: expected calibration error, Brier score, and accuracy at a fixed abstain rate. `run.py` would need those three metrics added, since today it only scores the top answer.

The upgrade rungs need a held-out split. One treasury case is 1.5 points of accuracy and one support case is 8 points, so the current 78 cases can show the zero-data result but are too few both to train a probe and to measure it.

## Open questions

- [ ] Does the 4-bit `mlx_vlm` checkpoint expose the final hidden state and logits from one forward call, and does Gemma 4 apply logit softcapping that the head must reproduce?
- [ ] How do the letters, digits, Yes and No tokenise in Gemma 4 (with or without a leading space), and is each a single token?
- [ ] How strong is Gemma's position bias on these suites, and is order averaging worth two or three times the compute?
- [ ] For `score` questions, should the answer be the most likely level or the expected value rounded?
- [ ] One pass per question, or all questions of a request in one pass with a separate answer position each?
- [ ] Where will labels come from in production: human review of abstained cases, or Jev's answers as a teacher, and do Jev's terms allow that?
- [ ] Is `gemma-4-e4b-it` (instruction-tuned) the right backbone for the probe rungs, or would the base checkpoint probe better?
- [ ] What abstain rate is acceptable, which sets the confidence threshold?
