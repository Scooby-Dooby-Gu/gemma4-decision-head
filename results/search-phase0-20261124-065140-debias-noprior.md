# Search POC phase 0: order-averaged, without the blank-input prior

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 10 forward passes per update. Model load 2.9 s; first query (prompts, priors, prefix caches) 3.1 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 646 ms | 662 ms | 637 ms | 669 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 76.1% | 50.2% |
| state_west | 95.5% | 90.8% |
| kids | 93.3% | 80.0% |
| age | 91.0% | 63.9% |
| marital | 97.8% | 91.5% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 83.6% | 98.5% | 92.5% | 94.0% | 94.0% | 97.8% | 72 of 305 (23.6%) | 33 | 2 of 201 |
| 0.7 | 85.8% | 97.8% | 93.3% | 95.5% | 95.5% | 98.5% | 55 of 305 (18.0%) | 29 | 2 of 201 |
| 0.9 | 85.8% | 97.8% | 93.3% | 94.8% | 94.8% | 100.0% | 29 of 305 (9.5%) | 22 | 7 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 86.7% (105) | 66.7% (15) | 78.6% (14) |
| 0.7 | 88.6% (105) | 66.7% (15) | 85.7% (14) |
| 0.9 | 88.6% (105) | 66.7% (15) | 85.7% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| the big apple | state=NY | state=MA | `state = 'MA'` |
| with kids | kids=has_kids | kids=has_kids, age=young, ages 18–34 | `kids > 0 AND age >= 18 AND age <= 34` |
| child free | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| widowed | marital=widowed | age=seniors, marital=widowed, ages 65– | `age >= 65 AND marital = 'widowed'` |
| over 40 | age=older_than, ages 41– | kids=has_kids, age=older_than, ages 41– | `kids > 0 AND age >= 41` |
| 50+ | age=older_than, ages 50– | kids=has_kids, age=older_than, ages 50– | `kids > 0 AND age >= 50` |
| 40 and up | age=older_than, ages 40– | kids=has_kids, age=older_than, ages 40– | `kids > 0 AND age >= 40` |
| thirtysomethings | age=decade, ages 30–39 | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| young professionals | age=young, ages 18–34 | age=young, marital=single, ages 18–34 | `age >= 18 AND age <= 34 AND marital = 'single'` |
| families in boston | state=MA, kids=has_kids | state=MA | `state = 'MA'` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, marital=married | `state = 'CO' AND kids = 0 AND marital = 'married'` |
| parents in tennessee | state=TN, kids=has_kids | state=TN | `state = 'TN'` |
| with kids over 10 | kids=has_kids | kids=has_kids, age=older_than, ages 11– | `kids > 0 AND age >= 11` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| solteros menores de 30 | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| padres divorciados | kids=has_kids, marital=divorced | marital=divorced | `marital = 'divorced'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| en sus cuarentas | age=decade, ages 40–49 | (none) | `TRUE` |
| sin niños en nuevo méxico | state=NM, kids=no_kids | state=NM, kids=has_kids, age=young, ages 18–34 | `state = 'NM' AND kids > 0 AND age >= 18 AND age <= 34` |
| veufs | marital=widowed | age=seniors, marital=widowed, ages 65– | `age >= 65 AND marital = 'widowed'` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA | `state = 'MA'` |
| familles à seattle | state=WA, kids=has_kids | state=WA | `state = 'WA'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| people | kids=has_kids |
| clients in washington | state=WA |
| people | kids=has_kids |
| the bay | state=WA |
| anyone in the lone | marital=single |
| anyone in the lone star | marital=single |
| child | kids=has_kids, age=young |
| over | kids=has_kids |
| over | kids=has_kids |
| under | kids=has_kids, marital=single |
| under | kids=has_kids, marital=single |
| 40 | state=CO, kids=has_kids |
| 40 and | kids=has_kids |
| in | kids=has_kids |
| in | kids=has_kids |
| 40 | state=CO, kids=has_kids, age=older_than |
| 40 year | age=older_than |
| new york with kids | age=young |
| new | kids=has_kids |
| child | kids=has_kids, age=young |
| child free | marital=single |
| child free in | marital=single |
| widowed | age=seniors |
| widowed over | age=seniors |
| between 30 and 45 in | kids=has_kids |
| between 30 and 45 in new | kids=has_kids |
| 50+ | kids=has_kids |
| 50+ and | kids=has_kids |
| young professionals | marital=single |
| young professionals in | marital=single |
| single under | kids=no_kids |
| people | kids=has_kids |
| people over | kids=has_kids |
| married with children in new | state=NM |
| widows | age=seniors |
| widows in | age=seniors |
| childless couples | marital=married |
| childless couples in | marital=married |
| moms in | marital=single |
| clients in north carolina aged 25 | age=younger_than |
| new | kids=has_kids |
| young parents | marital=single |
| young parents in | marital=single |
| with kids | age=young |
| clients with 3 | age=older_than, marital=single |
| new | kids=has_kids |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| anyone who mentioned | kids=has_kids |
| solteros | kids=no_kids |
| solteros menores | kids=no_kids |
| solteros menores de | kids=no_kids |
| personas | kids=has_kids |
| personas de 40 a | kids=has_kids |
| viudas en | age=seniors |
| en | kids=has_kids |
| sin | marital=single |
| sin niños | kids=has_kids, age=young |
| sin niños en | kids=has_kids, age=young |
| sin niños en nuevo | kids=has_kids, age=young |
| mayores | kids=has_kids |
| mayores de | kids=has_kids |
| jóvenes | kids=has_kids, marital=single |
| jóvenes en | kids=has_kids, marital=single |
| madres solteras en nueva | state=NM |
| célibataires | kids=no_kids |
| célibataires moins | kids=no_kids |
| célibataires moins de | kids=no_kids |
| célibataires moins de 30 | kids=no_kids |
| la | state=LA |
| jeunes | kids=has_kids, marital=single |
| jeunes à | kids=has_kids, marital=single |
