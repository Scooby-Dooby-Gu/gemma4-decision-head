# Search POC, phase 0: can the decision head turn a search box into filters as you type?

The idea is from a CRM where typing "new york with kids over 40" sets a state filter, then a kids filter, then an age filter, each as soon as its words are typed. The model never reads the client records. It turns the query into filters, and the database does the filtering.

Phase 0 tests the two risks before building any UI: whether the head is fast enough to re-run on every word, and whether it stays quiet on half-typed queries.

## Setup

- **Five questions** in [`filters.py`](filters.py): `state_east` (26 states + DC) and `state_west` (24 states), split because 51 options is more than a question can have; `kids`; `age`; `marital`. Each has a "none" or "not mentioned" option. The head picks the *kind* of age filter ("older than", "in their forties", "retirees") and a small parser reads the number from the text, in English, Spanish and French.
- **A filter is applied** when the head picks a value, doesn't abstain, and its confidence is at least a threshold.
- **134 hand-written queries** in [`queries.txt`](queries.txt) (105 English, 15 Spanish, 14 French): synonyms, nicknames ("the big apple", "nola"), lookalikes ("new mexico", "west virginia"), and traps where nothing or only some filters should appear ("with kids over 10", "new leads", "john smith").
- **Each query is typed a word at a time**, so 439 updates. Until a word finishes a segment, that segment's filters are expected to be off.
- Zero labels: no head is trained. `mlx-community/gemma-4-e4b-it-4bit` on an M5 MacBook.

Run it with `python -m poc.search.measure` (flags `--no-debias`, `--no-prior`, `--limit`, `--report`).

## Results

Four variants: with and without order averaging (two passes per question in listed and reversed order, or one), and with and without the blank-input prior (scoring each question on "N/A" and dividing out 0.7 of that answer).

| Variant | Passes per update | p50 / p95 per update | All filters right (0.7 / 0.9) | Wrong filter while typing (0.7 / 0.9) |
|---|---|---|---|---|
| Order-averaged, with prior (repo default) | 10 | 638 / 655 ms | 77.6% / 81.3% | 31.5% / 17.0% |
| Order-averaged, no prior | 10 | 646 / 662 ms | 85.8% / 85.8% | 18.0% / 9.5% |
| Listed order, with prior | 5 | 320 / 328 ms | 82.1% / 77.6% | 28.5% / 15.1% |
| **Listed order, no prior** | **5** | **315 / 326 ms** | **85.8% / 86.6%** | **16.1% / 10.8%** |

*All filters right* is the share of finished queries where every filter matches. *Wrong filter while typing* is the share of 305 partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

Best variant (listed order, no prior), per filter at threshold 0.9: state 96%, kids 95%, age 96%, marital 100%. By language: English 90%, French 86%, Spanish 67% (15 queries, so one query is 7 points).

Full reports, including every failure, are in `results/search-phase0-20261124-*.md`; the raw answers are in the matching `.json`, so other thresholds can be scored with `--report` without the model.

## What this says

**1. The blank-input prior is wrong for search.** It is the biggest effect in the table. On a blank input Gemma answers "not mentioned" almost certainly, so dividing that out pushes every answer away from "not mentioned", which is exactly the option a search filter needs most. Turning it off took finished-query accuracy from 63% to 84% at threshold 0.5 (order-averaged) and halved the wrong filters while typing. This is the same over-correction the design doc found for state-first prompts.

**2. Order averaging isn't worth its cost here.** Without the prior, listed order alone matches it on accuracy and halves latency.

**3. ~320 ms is not "instant".** Each forward pass costs about 63 ms even though only the query's ~10 tokens are new, and five filters means five passes. That's usable with a debounce, but it's not the "filtered before you look down" the CRM comment describes. More filters make it linearly worse. Phase 2's one-pass idea (encode the query once, read every filter from that hidden state with trained probes) is the way to get under 100 ms, and it now looks necessary rather than optional.

**4. Most wrong filters while typing are the model inferring things nobody asked for.** At 0.9, 16 of the 34 wrong filters are kids and 11 are marital: "solteros" / "célibataires" → no kids (English "single" doesn't do this), "young professionals" / "jeunes" → single, "child" alone → has kids, "40" alone → has kids. Training on labelled prefixes ("solteros" → kids not mentioned) is the direct fix.

**5. The misses on finished queries cluster in a few places:**
- *Group words for kids*: "families", "familias", "familles", "padres" are never confidently has-kids.
- *Age words without a digit*: "thirtysomethings", "en sus cuarentas", "la quarantaine" stay off; "0 kids under 35" misses the age.
- *States*: "virginia" and "west virginia" stay off, "nola" stays off, "the big apple" → Massachusetts. The east question (28 options with "none") is a little weaker than the west one (25): 94% vs 96% top-option accuracy on finished queries.
- *Traps it falls for*: "with kids over 10" sets age over 10 for the client, and "clients with a georgia phone number" sets Georgia (arguably reasonable).

All of these are what 20 labels per option fixed on the intent benchmarks, so they look like a training problem, not a ceiling.

## Caveats

- 134 queries written by me, labelled by me. One label was wrong in the first run ("married folks in the bay area" didn't expect `marital=married`); it's fixed and the runs above are after the fix. "childless couples" doesn't expect `marital=married`, which is debatable.
- One backbone, one machine, zero labels, no comparison yet against a generative model producing JSON filters.
- Latency is the model only: no HTTP, no database.

## Next

- Default to listed order and no prior for search questions.
- Phase 1 as planned (fake CRM in SQLite, FastAPI, the search page, a side-by-side against Ollama `gemma4:12b` producing JSON), using this setup.
- Before phase 2, label prefixes and the weak spots above (about 20 per option) and re-run this measurement with `train()`, to see how much of the wrong-while-typing problem labels remove.
- Try the one-pass, many-probes layout for latency.

# Phase 1: the demo, and the 12B side by side

```bash
python -m poc.search.app            # http://127.0.0.1:8093 (needs Ollama with gemma4:12b-mlx for the comparison)
python -m poc.search.app --no-llm   # decision head only
```

- [`crm.py`](crm.py) builds `crm.db`: 5,000 made-up clients (seeded, names from short lists paired at random) with state, city, age, kids, marital status, occupation and last contact. It's rebuilt if missing and not committed. A filtered query takes under 1 ms, so the model is the whole wait.
- [`app.py`](app.py) is a stdlib HTTP server like `serve.py`. `/api/head` runs the decision head in the phase 0 setup (listed order, no blank-input prior). `/api/llm` asks `gemma4:12b-mlx` through Ollama to write the same filters as JSON. Both run the same parameterised SQL.
- [`index.html`](index.html) is the search page: filters update on each finished word and after a 250 ms pause; chips show confidence; dashed chips are what the head leans towards but hasn't applied; × removes a chip until that filter's value changes; a threshold menu; and a Both mode that runs the two engines side by side.

Ollama's MLX backend refuses structured output ("structured output is unavailable"), so the 12B is asked for JSON in the prompt and the reply is parsed. It replied with valid JSON every time in the run below.

## Decision head vs 12B writing JSON

[`compare.py`](compare.py) runs the 12B over the same 439 typed prefixes and scores both the same way, with ages compared as ranges because the 12B writes a range. Full report: `results/search-compare-20261124-082322.md`.

| | Decision head (E4B, threshold 0.9) | 12B writes JSON |
|---|---|---|
| Latency per update, p50 / p95 | **315 / 326 ms** | 940 / 1,125 ms |
| All filters right | 86.6% | **92.5%** |
| state / kids / age / marital | 95.5 / 94.8 / 96.3 / 100% | 99.3 / 99.3 / 95.5 / 98.5% |
| English / Spanish / French | 89.5 / 66.7 / 85.7% | 93.3 / 86.7 / 92.9% |
| Wrong filter while typing | 10.8% | 12.1% |

With no labels the 12B is more accurate, and the gap is mostly states and kids: it knows "nola", "the big apple" and "virginia", and reads "families" and "padres" as having kids (though it missed "parents" on its own). Its misses are mostly about where an age range starts and ends ("over 65" → 65, "around 50" → exactly 50, "under 30" → 0–29, which returns the same rows), plus two it shares with the head ("with kids over 10", "georgia phone number"). The head is three times faster on a model a third the size. Neither one stays quiet on half-typed queries any better than the other.

So the commenter's case for a decision head rests on speed, and here that holds. On accuracy it holds only once the head is trained: every one of the head's misses above is the kind of thing 20 labels per option fixed on the intent benchmarks. Two other things to keep in mind:
- 940 ms is a 12B on a laptop. A hosted Flash-class model would be faster per token, but adds a network round trip, which is the lag the commenter describes.
- In Both mode the two models share the GPU, so each is slower than when it runs alone. The numbers in this table were measured one engine at a time.

# Labelling: training the heads

```bash
python -m poc.search.labels         # writes train_cases.json (1,690 cases)
python -m poc.search.train_heads    # ~9 min of backbone passes the first time, then seconds
python -m poc.search.measure --no-debias --no-prior --heads poc/search/heads
python -m poc.search.app --heads poc/search/heads
```

Hand-writing 20 queries for each of ~60 options isn't practical, so [`labels.py`](labels.py) has a pool of short phrases per option in English, Spanish and French (cities, nicknames, "who have kids", "mayores de {n} años", "dans la quarantaine") and combines them with filler words into 715 queries, at least 10 per state and 36 or more per kids, age or marital option. About half of each query's word-by-word prefixes are added too, for 1,690 cases in all. Heads go in `poc/search/heads/` and hidden states in `poc/search/cache/`; neither is committed.

**Kept apart from the test set.** No training query or prefix is the same text as a test query or prefix (a test checks this). No phrase the test set labels is picked from the pools ("families", "padres", "the big apple", "nola", "dads", "seattle"), except a state's own English name, which is the option's own wording. There are no look-alikes of the test traps. So the test still measures phrasing the head hasn't seen.

**A labelling mistake worth knowing about.** The first run labelled each prefix with only the phrases finished so far, as the test does. That's fine for scoring and wrong for training: "married" is a prefix of the training phrase "married couples", so it was labelled "not mentioned", and so were "seniors" (from "seniors à la retraite") and "more than 40" (from "more than 40 years old"). The heads learned to ignore complete filters, and finished-query accuracy fell to 47.8% at threshold 0.9 (`results/search-phase0-20261124-084208-listed-noprior-trained-mislabelled.md`). Now a prefix that stops partway through a phrase gives that phrase's question no label and still labels the others.

## Results

All five heads trained at rung 2 (bias, temperature and per-option correction), chosen by the guard on a fifth of the labels held out. Trained heads are calibrated, so they're less sure of themselves, and 0.5 is now the best threshold (it was 0.9 with zero labels). Full reports: `results/search-phase0-20261124-084530-listed-noprior-trained.md` and `results/search-compare-20261124-084534.md`.

| | Head, zero labels (0.9) | Head, trained (0.5) | 12B writes JSON |
|---|---|---|---|
| Latency per update, p50 | 315 ms | 321 ms | 940 ms |
| All filters right | 86.6% | **90.3%** | **92.5%** |
| state / kids / age / marital | 95.5 / 94.8 / 96.3 / 100% | 97.8 / 96.3 / 97.8 / 98.5% | 99.3 / 99.3 / 95.5 / 98.5% |
| English / Spanish / French | 89.5 / 66.7 / 85.7% | 91.4 / 86.7 / 85.7% | 93.3 / 86.7 / 92.9% |
| Wrong filter while typing | 10.8% | **5.6%** | 12.1% |

Ages compared as ranges in all three columns, so the zero-label figures differ slightly from phase 0's table.

- **Training closed most of the gap.** 90.3% against the 12B's 92.5%, at a third of the latency, and Spanish went from 66.7% to 86.7%.
- **It halved the wrong filters while typing,** and it's now well under the 12B's rate. "solteros" alone no longer sets no kids, "jeunes" no longer sets single.
- **What's left is mostly unseen phrasing.** "families", "familias", "familles", "padres" and "parents in tennessee" still don't set has-kids, and "the big apple" is still Massachusetts. All of those were kept out of training on purpose; a real deployment would label them. Also wrong: "married no kids over 35" read as under 35, "la quarantaine" as middle-aged (40–59 rather than 40–49), "not married" left off, and "leads in washington state" left off.
- **The two traps both engines fall for are still there:** "with kids over 10" and "georgia phone number".

Caveats: the training phrases and the test queries were both written by me, so they share a style even where the words differ; the quiz the guard uses comes from the same generated phrases, so its losses are optimistic; one seed of generation and training.

## Next

1. **Real phrasing.** Labels from people actually searching (or another person writing them) would say more than my pools do. The misses above are the list to start from.
2. **One pass for all filters.** Encode the query once and read every filter from that hidden state with trained probes, to get from ~315 ms to under 100 ms.

# Jev: TypeSafe's hosted decision model

```bash
# a key in .env.local (gitignored): OPENROUTER_API_KEY=... or TYPESAFE_API_KEY=...
python -m poc.search.jev --name-keys          # 439 updates, about 2 minutes, about $0.03
python -m poc.search.compare results/search-phase0-20261124-084530-listed-noprior-trained.json \
    --threshold 0.5 --llm-run results/search-llm12b-20261124-082322.json \
    --jev-run results/search-jev-20261124-201524-namekeys.json
```

[`jev.py`](jev.py) sends Jev (`typesafe/jev-1.14-20261014`, through OpenRouter's Decisions API) the same five questions the head gets, for every word-by-word prefix of the same 134 test queries. It scores the replies with the same `to_filters`. Each update is one request of about 1,450 input tokens, almost all of them the questions, and the whole run cost under 3 cents. Jev's times are end to end from this laptop, network included. The head and the 12B run locally.

**State keys matter to Jev.** The first run used the head's questions unchanged, with states keyed by two-letter code. Jev reads an option's key as well as its description, so the word "in" became Indiana, "la" Louisiana and "pas" Pennsylvania while the query was still being typed (`results/search-jev-20261124-201344.json`). The head never sees the keys. Keying states by name (`--name-keys`: "indiana", "new_york") removed those, so that's the run in the table.

| | Head, trained (0.5) | 12B writes JSON | Jev, state codes (0.5) | Jev, state names (0.5) |
|---|---|---|---|---|
| Latency per update, p50 / p95 | 321 / 339 ms | 940 / 1,125 ms | 139 / 193 ms | **137 / 189 ms** |
| All filters right | 90.3% | **92.5%** | 91.8% | **92.5%** |
| state / kids / age / marital | 97.8 / 96.3 / 97.8 / 98.5% | 99.3 / 99.3 / 95.5 / 98.5% | 99.3 / 94.8 / 98.5 / 99.3% | 99.3 / 94.8 / 99.3 / 99.3% |
| English / Spanish / French | 91.4 / 86.7 / 85.7% | 93.3 / 86.7 / 92.9% | 92.4 / 86.7 / 92.9% | 92.4 / 93.3 / 92.9% |
| Wrong filter while typing | **5.6%** | 12.1% | 11.1% | **5.6%** |

Full report: `results/search-compare-20261124-201537.md`.

Jev's results depend on the threshold much more than the trained head's do. The head's answers are almost never below 0.6 confidence, so its row barely moves:

| Threshold | 0.0 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| Head, trained: all right / wrong while typing | 90.3 / 5.6% | 90.3 / 5.6% | 90.3 / 4.9% | 88.1 / 4.9% | 86.6 / 4.3% | 82.8 / 3.0% |
| Jev, state names: all right / wrong while typing | 96.3 / 10.8% | 92.5 / 5.6% | 91.0 / 4.3% | 88.8 / 3.6% | 88.1 / 3.0% | 83.6 / 2.0% |

- **Jev is the fastest of the three, network included.** About 140 ms end to end against 321 ms for the head on this laptop. That's the "instant" the commenter describes, and it holds with five questions in one request. The head's time is five passes. Reading every filter from one pass (Next, item 2) is the change that would close that gap.
- **With no tuning, Jev is as accurate as the 12B.** 92.5% at 0.5, and 96.3% with no threshold, where it gets every Spanish query right. Jev's no-threshold score is the best finished-query score of anything tested here. It comes at 10.8% wrong while typing, though; 0.5 or 0.6 gives a better balance.
- **Its misses are mostly words that imply children.** "parents", "moms", "families in boston", "texas parents over 50", "parents in tennessee", "padres divorciados" and "parents divorcés" don't set has-kids. For "parents" Jev picks "not mentioned" at 0.35. It seems to read the kids question literally. Rewording the option ("Wants clients who have children, such as parents") would probably fix it, but that would be tuning to the test set, so it isn't done here. Unlike the head, Jev can't be trained on labels: that wording is the only lever.
- **The shared traps:** "georgia phone number" catches all three, and "childless couples" reads as married for Jev and the head.

Caveats: 0.5 was picked on the test set, for Jev as for the head. One run, from one place at one time of day. A hosted model can change, so the run pins `jev-1.14`. The kids misses are phrasing kept out of the head's training, and Jev never saw any of my phrasing at all.
