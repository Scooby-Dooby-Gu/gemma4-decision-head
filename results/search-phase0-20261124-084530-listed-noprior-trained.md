# Search POC phase 0: listed order only, without the blank-input prior, trained heads

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 5 forward passes per update. Model load 2.2 s; first query (prompts, priors, prefix caches) 1.5 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 321 ms | 339 ms | 322 ms | 340 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 98.5% | 99.0% |
| state_west | 99.3% | 99.0% |
| kids | 96.3% | 97.0% |
| age | 97.8% | 84.9% |
| marital | 98.5% | 98.7% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 90.3% | 97.8% | 96.3% | 97.8% | 97.8% | 98.5% | 17 of 305 (5.6%) | 19 | 4 of 201 |
| 0.7 | 88.1% | 97.0% | 94.0% | 98.5% | 98.5% | 98.5% | 15 of 305 (4.9%) | 18 | 7 of 201 |
| 0.9 | 82.8% | 93.3% | 93.3% | 96.3% | 96.3% | 98.5% | 9 of 305 (3.0%) | 16 | 18 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 91.4% (105) | 86.7% (15) | 85.7% (14) |
| 0.7 | 90.5% (105) | 80.0% (15) | 78.6% (14) |
| 0.9 | 85.7% (105) | 73.3% (15) | 71.4% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| leads in washington state | state=WA | (none) | `TRUE` |
| the big apple | state=NY | state=MA | `state = 'MA'` |
| not married | marital=single | (none) | `TRUE` |
| families in boston | state=MA, kids=has_kids | state=MA | `state = 'MA'` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, marital=married | `state = 'CO' AND kids = 0 AND marital = 'married'` |
| married no kids over 35 | kids=no_kids, age=older_than, marital=married, ages 36– | kids=no_kids, age=younger_than, marital=married, ages –34 | `kids = 0 AND age <= 34 AND marital = 'married'` |
| parents in tennessee | state=TN, kids=has_kids | state=TN | `state = 'TN'` |
| with kids over 10 | kids=has_kids | kids=has_kids, age=older_than, ages 11– | `kids > 0 AND age >= 11` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| padres divorciados | kids=has_kids, marital=divorced | marital=divorced | `marital = 'divorced'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA, age=middle_aged, ages 40–59 | `state = 'MA' AND age >= 40 AND age <= 59` |
| familles à seattle | state=WA, kids=has_kids | state=WA | `state = 'WA'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| anyone in the lone | marital=single |
| 40 | age=around |
| 40 and | age=around |
| aged 35 | age=around |
| younger | age=young |
| childless couples | marital=married |
| childless couples in | marital=married |
| clients in north carolina aged 25 | age=around |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| solteros menores | age=young |
| solteros menores de | age=young |
| personas de 40 | age=decade |
| entre 35 | age=around |
| la | state=LA |
| la quarantaine | age=middle_aged |
| la quarantaine à | age=middle_aged |
