# Search POC phase 0: listed order only, without the blank-input prior, trained heads

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 5 forward passes per update. Model load 2.7 s; first query (prompts, priors, prefix caches) 1.5 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 330 ms | 340 ms | 326 ms | 346 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 98.5% | 99.3% |
| state_west | 99.3% | 98.0% |
| kids | 94.0% | 92.8% |
| age | 90.3% | 94.8% |
| marital | 97.0% | 95.1% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 76.9% | 97.8% | 93.3% | 89.6% | 89.6% | 96.3% | 9 of 305 (3.0%) | 9 | 46 of 201 |
| 0.7 | 67.2% | 97.8% | 88.1% | 86.6% | 86.6% | 94.0% | 8 of 305 (2.6%) | 6 | 60 of 201 |
| 0.9 | 47.8% | 95.5% | 79.1% | 79.1% | 79.1% | 89.6% | 8 of 305 (2.6%) | 2 | 96 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 78.1% (105) | 73.3% (15) | 71.4% (14) |
| 0.7 | 68.6% (105) | 53.3% (15) | 71.4% (14) |
| 0.9 | 48.6% (105) | 40.0% (15) | 50.0% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| leads in washington state | state=WA | (none) | `TRUE` |
| the big apple | state=NY | state=MA | `state = 'MA'` |
| with kids | kids=has_kids | (none) | `TRUE` |
| moms | kids=has_kids | (none) | `TRUE` |
| 0 kids | kids=no_kids | (none) | `TRUE` |
| married | marital=married | (none) | `TRUE` |
| divorced | marital=divorced | (none) | `TRUE` |
| not married | marital=single | (none) | `TRUE` |
| between 30 and 45 | age=between, ages 30–45 | (none) | `TRUE` |
| aged 35 to 50 | age=between, ages 35–50 | (none) | `TRUE` |
| 40 year olds | age=around, ages 38–42 | (none) | `TRUE` |
| seniors | age=seniors, ages 65– | (none) | `TRUE` |
| new york no kids | state=NY, kids=no_kids | state=NY | `state = 'NY'` |
| widowed over 60 | age=older_than, marital=widowed, ages 61– | marital=widowed | `marital = 'widowed'` |
| families in boston | state=MA, kids=has_kids | state=MA | `state = 'MA'` |
| between 30 and 45 in new jersey | state=NJ, age=between, ages 30–45 | state=NJ | `state = 'NJ'` |
| 50+ and married | age=older_than, marital=married, ages 50– | age=older_than, ages 50– | `age >= 50` |
| texas parents over 50 | state=TX, kids=has_kids, age=older_than, ages 51– | state=TX, kids=has_kids | `state = 'TX' AND kids > 0` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, marital=married | `state = 'CO' AND kids = 0 AND marital = 'married'` |
| clients in north carolina aged 25 to 40 | state=NC, age=between, ages 25–40 | state=NC | `state = 'NC'` |
| single without kids | kids=no_kids, marital=single | marital=single | `marital = 'single'` |
| new hampshire clients over 60 | state=NH, age=older_than, ages 61– | state=NH | `state = 'NH'` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| padres divorciados | kids=has_kids, marital=divorced | marital=divorced | `marital = 'divorced'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| mayores de 65 | age=older_than, ages 66– | (none) | `TRUE` |
| jóvenes en miami | state=FL, age=young, ages 18–34 | state=FL | `state = 'FL'` |
| entre 35 et 50 ans | age=between, ages 35–50 | (none) | `TRUE` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA, age=middle_aged, ages 40–59 | `state = 'MA' AND age >= 40 AND age <= 59` |
| familles à seattle | state=WA, kids=has_kids | state=WA | `state = 'WA'` |
| jeunes à chicago | state=IL, age=young, ages 18–34 | state=IL | `state = 'IL'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| anyone in the lone | marital=single |
| childless couples | marital=married |
| childless couples in | marital=married |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| personas de 40 | age=decade |
| la | state=LA |
| la quarantaine | age=middle_aged |
| la quarantaine à | state=LA, age=middle_aged |
