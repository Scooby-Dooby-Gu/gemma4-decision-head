# Search POC phase 0: listed order only, without the blank-input prior

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 5 forward passes per update. Model load 2.5 s; first query (prompts, priors, prefix caches) 1.6 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 315 ms | 326 ms | 311 ms | 327 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 94.0% | 88.5% |
| state_west | 96.3% | 90.8% |
| kids | 94.8% | 79.0% |
| age | 90.3% | 65.9% |
| marital | 97.8% | 90.5% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 84.3% | 97.0% | 95.5% | 94.0% | 94.0% | 97.8% | 75 of 305 (24.6%) | 35 | 2 of 201 |
| 0.7 | 85.8% | 96.3% | 94.8% | 96.3% | 96.3% | 98.5% | 49 of 305 (16.1%) | 29 | 4 of 201 |
| 0.9 | 86.6% | 95.5% | 94.8% | 96.3% | 96.3% | 100.0% | 33 of 305 (10.8%) | 22 | 6 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 85.7% (105) | 73.3% (15) | 85.7% (14) |
| 0.7 | 87.6% (105) | 73.3% (15) | 85.7% (14) |
| 0.9 | 89.5% (105) | 66.7% (15) | 85.7% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| contacts in west virginia | state=WV | (none) | `TRUE` |
| the big apple | state=NY | state=MA | `state = 'MA'` |
| nola | state=LA | state=FL | `state = 'FL'` |
| moms | kids=has_kids | kids=has_kids, age=middle_aged, ages 40–59 | `kids > 0 AND age >= 40 AND age <= 59` |
| child free | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| 50+ | age=older_than, ages 50– | kids=has_kids, age=older_than, ages 50– | `kids > 0 AND age >= 50` |
| 40 and up | age=older_than, ages 40– | kids=has_kids, age=older_than, ages 40– | `kids > 0 AND age >= 40` |
| thirtysomethings | age=decade, ages 30–39 | age=around, ages 28–32 | `age >= 28 AND age <= 32` |
| young professionals | age=young, ages 18–34 | age=young, marital=single, ages 18–34 | `age >= 18 AND age <= 34 AND marital = 'single'` |
| 0 kids under 35 | kids=no_kids, age=younger_than, ages –34 | kids=no_kids | `kids = 0` |
| families in boston | state=MA, kids=has_kids | state=MA | `state = 'MA'` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, marital=married | `state = 'CO' AND kids = 0 AND marital = 'married'` |
| with kids over 10 | kids=has_kids | kids=has_kids, age=older_than, ages 11– | `kids > 0 AND age >= 11` |
| vip clients | (none) | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| solteros menores de 30 | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| en sus cuarentas | age=decade, ages 40–49 | (none) | `TRUE` |
| sin niños en nuevo méxico | state=NM, kids=no_kids | state=NM, kids=no_kids, age=young, ages 18–34 | `state = 'NM' AND kids = 0 AND age >= 18 AND age <= 34` |
| célibataires moins de 30 ans | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA | `state = 'MA'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| people | kids=has_kids |
| clients in washington | state=WA |
| people | kids=has_kids |
| the big | age=middle_aged |
| the bay | state=MD |
| anyone in the lone | marital=single |
| anyone in the lone star | marital=single |
| child | kids=has_kids |
| over | kids=has_kids |
| over | kids=has_kids |
| under | kids=no_kids |
| under | kids=no_kids |
| 40 | kids=has_kids |
| aged 35 | age=around |
| in | kids=has_kids |
| in | kids=has_kids |
| 40 | kids=has_kids |
| young | marital=single |
| new york with kids | age=young |
| new | kids=has_kids |
| child | kids=has_kids |
| child free | marital=single |
| child free in | marital=single |
| widowed over | age=seniors |
| between 30 and 45 in | kids=has_kids |
| 50+ | kids=has_kids |
| 50+ and | kids=has_kids |
| young | marital=single |
| young professionals | marital=single |
| young professionals in | marital=single |
| single under | kids=no_kids |
| people | kids=has_kids |
| people over | kids=has_kids |
| married with children in new | state=NM |
| widows | age=seniors |
| widows in | age=seniors |
| single in | kids=no_kids |
| childless couples | marital=married |
| childless couples in | marital=married |
| moms | age=middle_aged |
| moms in | age=middle_aged |
| clients in north carolina aged 25 | age=younger_than |
| new | kids=has_kids |
| young | marital=single |
| young parents | marital=single |
| young parents in | marital=single |
| clients who | kids=has_kids |
| top | kids=has_kids |
| vip | age=middle_aged |
| new | kids=has_kids |
| clients with | kids=has_kids |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| anyone who mentioned | kids=has_kids, marital=single |
| solteros | kids=no_kids |
| solteros menores | kids=no_kids |
| solteros menores de | kids=no_kids |
| jubilados | marital=widowed |
| personas | kids=has_kids |
| personas de 40 a | kids=has_kids |
| en | kids=has_kids |
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
| la quarantaine | age=older_than |
| la quarantaine à | age=older_than |
| jeunes | kids=has_kids, marital=single |
| jeunes à | kids=has_kids, marital=single |
