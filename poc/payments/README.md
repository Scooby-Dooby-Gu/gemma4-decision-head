# Card transaction search: the same test on payments

The client search in `poc/search` was rerun on a second domain: typed search over card transactions. "gas over $50 last month", "fraude el mes pasado" and "vols de plus de 500 dollars" become filters, updated after every word. Same method, same four models, harder filters: money amounts and dates, not just ages, and seven questions per update instead of five.

## Data

[Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation) (MIT) generates synthetic card transactions. It ran at commit `b5eb45c` with 300 cardholders, seed 7, from 2026-01-01 to 2026-11-24. That gave 227,468 transactions in 14 categories across 43 states, 2,798 of them fraud. Sparkov runs in its own folder and virtual environment (`.vendor/sparkov`, not committed). [`data.py`](data.py) loads its files into `txn.db` and leaves out the fake SSNs, card numbers and addresses it writes. The steps are in `data.py`.

## The demo

```bash
python -m poc.payments.app        # http://127.0.0.1:8094
```

A search box over the transactions that updates its filters after every word. Pick the trained Gemma head, trained Laya or Jev, or Compare to see all three side by side. Each applied filter is a chip with its confidence; × removes it until its value changes. Dashed chips show what an engine leans towards but hasn't applied, such as an amount not typed yet. Under each engine are its time, the SQL `WHERE` clause and the number of matches, and the newest matching transactions are listed below.

- **What it loads:** the trained Gemma heads from `poc/payments/heads/`, and Laya from `poc/payments/laya/checkpoint`. Laya runs in [`laya_server.py`](laya_server.py) in Laya's own environment, which [`app.py`](app.py) starts. Jev needs a key in `.env.local`. An engine whose files or key are missing is left out.
- **Launch entry:** `payments-search`.

**A fix to the data.** Sparkov at `b5eb45c` draws each transaction's date, time of day and category from linked random columns and then sorts them. As a result, each category landed in a few months: almost all gas in January and February, groceries in March, dining in July and August, travel in November. [`data.py`](data.py)'s `spread_dates` shuffles each cardholder's dates among their legitimate transactions, with a fixed seed, so every month has a normal mix. Monthly totals and fraud dates are unchanged. The benchmark only scores filters, not the rows they return, so its results don't depend on this.

## Filters

[`filters.py`](filters.py) asks seven questions, each with a "not mentioned" option:

| Question | Options |
|---|---|
| category | 10 spending categories (gas and transport, groceries, shopping, dining, entertainment, health and fitness, home, kids and pets, personal care, travel) |
| channel | online, in person |
| amount | over, under, between, around (±10%), large ($500+), small (up to $20) |
| date | a named month, a named weekday, weekends, today, yesterday, past week, this month, last month |
| state_east, state_west | the cardholder's state, as in the client search |
| fraud | fraud, legitimate only |

The head picks the kind of amount or date filter. A small parser reads the number, month or weekday from the text. Numbers marked as money win, so "orders for 2 people over $50" reads $50. Relative dates count from 2026-11-24, a Tuesday and the last day of the data.

## Test set and training examples

- **Test set:** [`queries.txt`](queries.txt) has 144 queries: 102 English, 21 Spanish, 21 French. Typed word by word, that's 397 updates. It includes traps: "charges that may be fraud" (not the month May), "the last 3 purchases", "all transactions in 2026", and words like "in", "la" and "pas" that match state codes.
- **Training examples:** [`labels.py`](labels.py) generates 1,887 of them with the same method and the same separation from the test set as the client search. No training text equals a test query or prefix, and no phrase the test labels is used. A test checks this.
- **Commands:**

```bash
python -m poc.payments.labels && python -m poc.payments.train_heads   # ~14 min of backbone passes, then seconds
python -m poc.payments.bench head                                     # and: head --heads poc/payments/heads, llm, jev
python -m poc.payments.bench compare <runs...> --thresholds 0.9,0.5,0.5,0.5
```

## Results

Full report: `results/payments-compare-20261124-211225.md`.

| | Gemma head, no examples (0.9) | Gemma head, trained (0.5) | Gemma 4 12B writing JSON | Jev (0.5) |
|---|---|---|---|---|
| Time per update, p50 | 436 ms | 424 ms | 1,561 ms | **137 ms** |
| Finished queries with every filter right | 56.9% | 91.7% | **93.8%** | 91.7% |
| English / Spanish / French | 56.9 / 57.1 / 57.1% | 94.1 / 90.5 / 81.0% | 95.1 / 95.2 / 85.7% | 93.1 / 95.2 / 81.0% |
| Wrong filter while typing (of 253 half-typed) | 27.3% | **2.4%** | 5.5% | 8.3% |

- **Without examples, the Gemma head is much worse here than on client search** (56.9% against 86.6%). It fills in what the query doesn't say. "Gas station visits" gets in-person, "groceries" gets legitimate only, and "online purchases" gets the shopping category. Training fixed nearly all of that: channel went from 79.9% to 98.6% and fraud from 81.2% to 98.6%.
- **Trained, the Gemma head ties Jev and is the steadiest while typing.** It sets a wrong filter on 2.4% of half-typed queries, against 5.5% for the 12B and 8.3% for Jev.
- **The 12B is the most accurate and now the slowest,** at 1.6 seconds per update because its instructions are longer. Its misses are about arithmetic. "Past week" starts a day early, and "about 40 dollars" becomes up to $40. While typing, it acts on half a phrase: "between $50" becomes over $50, and "coffee under" becomes under $20.
- **Jev reads "purchases" and "orders" as the shopping category** ("online purchases", "in-store purchases", "compras en línea", "achats en ligne"). That's 8 of its 12 misses. Without those it would be the most accurate.
- **Time grows with the number of questions for the Gemma head and not for Jev.** Seven questions take 424 ms against five questions' 321 ms. Jev stays at 137 ms either way.
- **French is the weakest language for every model.** "essence" (gas), "courses" (groceries) and "hier" (yesterday) were each missed by at least two of the three. "vols" means both flights and thefts, and the trained head read "vols de plus de 500 dollars" as fraud.

Caveats: the same author wrote the test queries, the training phrases and the labels. Some labels are judgement calls: should "online purchases" set shopping, or "grocery store runs" in-person? The 0.5 and 0.9 thresholds were picked on the test results. Each number comes from one run.

# Laya: an open-weight decision model, run locally

[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Apache 2.0) is the open-weight model Runware serves as `runware:laya@1`. It's an encoder (ModernBERT for English, 421M parameters; mmBERT for 100+ languages, 322M) that answers every question in one forward pass, in the same request format as Jev. It ran on this laptop's GPU through PyTorch. The code is at commit `54c7286` in `.vendor/laya`, and the weights are `convaiinnovations/laya` revision `fd4f9db`. Neither is committed.

```bash
.vendor/laya/.venv/bin/python -m poc.payments.bench laya                     # Laya's router, as shipped
python -m poc.payments.laya_train                                            # the 1,887 examples as laya-train JSONL
.vendor/laya/.venv/bin/laya-train --data poc/payments/laya/train.jsonl \
    --base <the multilingual checkpoint folder> --out poc/payments/laya/checkpoint --head-max-len 256
.vendor/laya/.venv/bin/python -m poc.payments.bench laya --laya-checkpoint poc/payments/laya/checkpoint
```

Laya got the same seven questions as Jev, with states keyed by name. Its question budget was raised from 192 to 256 tokens, because the state questions have 25 to 28 options and Laya's docs advise care above about 20. Training used `laya-train`'s defaults (the published notebook's settings: 4 epochs, batch 64, 752 updates) on the multilingual checkpoint, with the same 1,887 examples the Gemma head learned from. A question still being typed was left out of its row. One trained checkpoint answers every language.

| | Time per update, p50 / p95 | Every filter right | English / Spanish / French | Wrong filter while typing |
|---|---|---|---|---|
| Laya, as shipped (0.9) | 75 / 175 ms | 26.4% | 26.5 / 28.6 / 23.8% | 9.5% |
| Laya, trained (0.5) | **75 / 78 ms** | 91.7% | 93.1 / **100** / 76.2% | 2.8% |
| Jev (0.5) | 137 / 211 ms | 91.7% | 93.1 / 95.2 / 81.0% | 8.3% |
| Gemma head, trained (0.5) | 424 / 433 ms | 91.7% | 94.1 / 90.5 / 81.0% | **2.4%** |
| Gemma 4 12B writing JSON | 1,561 / 2,035 ms | **93.8%** | 95.1 / 95.2 / 85.7% | 5.5% |

Full report: `results/payments-compare-20261125-082255.md`.

- **As shipped, Laya isn't usable for this.** At its best threshold it left plain category words unset ("flights", "haircuts", "pet supplies"), and at lower ones it set fraud on most queries. Its own model card says the same: "a fast base to specialise, not a zero-shot decision engine". Its router also sends short text it can't place to the multilingual checkpoint, which covered 376 of 397 updates. Falling back to English instead didn't help (23.6%). Gating on Laya's own `route` flag set almost no filters.
- **Trained, it ties Jev and the Gemma head at a fifth of the Gemma head's time.** It reached 91.7% at 75 ms per update, the fastest of everything tested, with the steadiest timing (p95 78 ms). It made almost as few mistakes while typing as the trained Gemma head (2.8% against 2.4%), and got all 21 Spanish queries right.
- **The threshold barely matters once trained.** Every cutoff from 0 to 0.8 gives 91.7%. Laya fits its calibration on 400 of the training examples.
- **Its misses are category words it never saw,** mostly in French ("essence", "coiffeur", "salle de sport" as entertainment, "vols" left unset), plus "pas de fraude" read as fraud and "on fridays" read as weekends.
- **Training took 10 hours of wall time here because the Mac slept overnight.** An epoch took about 16 minutes awake, so about an hour in all, against 14 minutes of backbone passes plus seconds of fitting for the Gemma head. Training rewrites all of Laya's encoder weights; the Gemma head leaves Gemma frozen.
