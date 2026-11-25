# Card transaction search: the same test on payments

The client search in `poc/search` was rerun on a second domain: typed search over card transactions. "gas over $50 last month", "fraude el mes pasado" and "vols de plus de 500 dollars" become filters, updated after every word. Same method, same four models, harder filters: money amounts and dates, not just ages, and seven questions per update instead of five.

## Data

[Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation) (MIT) generates synthetic card transactions. It ran at commit `b5eb45c` with 300 cardholders, seed 7, from 2026-01-01 to 2026-11-24. That gave 227,468 transactions in 14 categories across 43 states, 2,798 of them fraud. Sparkov runs in its own folder and virtual environment (`.vendor/sparkov`, not committed). [`data.py`](data.py) loads its files into `txn.db` and leaves out the fake SSNs, card numbers and addresses it writes. The steps are in `data.py`.

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
