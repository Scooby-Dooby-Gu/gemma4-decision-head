# Search POC phase 0: listed order only, with the blank-input prior

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 5 forward passes per update. Model load 2.7 s; first query (prompts, priors, prefix caches) 1.5 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 320 ms | 328 ms | 317 ms | 330 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 71.6% | 40.7% |
| state_west | 88.8% | 57.4% |
| kids | 85.1% | 51.1% |
| age | 77.6% | 42.0% |
| marital | 85.1% | 56.1% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 70.9% | 97.0% | 88.8% | 90.3% | 90.3% | 91.8% | 153 of 305 (50.2%) | 40 | 2 of 201 |
| 0.7 | 82.1% | 97.8% | 93.3% | 94.8% | 94.8% | 96.3% | 87 of 305 (28.5%) | 33 | 5 of 201 |
| 0.9 | 77.6% | 96.3% | 89.6% | 94.8% | 94.8% | 97.0% | 46 of 305 (15.1%) | 27 | 11 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 71.4% (105) | 66.7% (15) | 71.4% (14) |
| 0.7 | 84.8% (105) | 66.7% (15) | 78.6% (14) |
| 0.9 | 82.9% (105) | 60.0% (15) | 57.1% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| contacts in west virginia | state=WV | (none) | `TRUE` |
| the big apple | state=NY | state=MA | `state = 'MA'` |
| socal clients | state=CA | state=CA, age=middle_aged, ages 40–59 | `state = 'CA' AND age >= 40 AND age <= 59` |
| nola | state=LA | state=LA, kids=no_kids, age=middle_aged, ages 40–59 | `state = 'LA' AND kids = 0 AND age >= 40 AND age <= 59` |
| with kids | kids=has_kids | kids=has_kids, age=young, marital=single, ages 18–34 | `kids > 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |
| moms | kids=has_kids | kids=has_kids, age=middle_aged, ages 40–59 | `kids > 0 AND age >= 40 AND age <= 59` |
| child free | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| 0 kids | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| childless | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| single | marital=single | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| not married | marital=single | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| under 30 | age=younger_than, ages –29 | kids=no_kids, age=younger_than, ages –29 | `kids = 0 AND age <= 29` |
| under 25 | age=younger_than, ages –24 | kids=no_kids, age=younger_than, ages –24 | `kids = 0 AND age <= 24` |
| 50+ | age=older_than, ages 50– | kids=has_kids, age=older_than, ages 50– | `kids > 0 AND age >= 50` |
| 40 and up | age=older_than, ages 40– | kids=has_kids, age=older_than, ages 40– | `kids > 0 AND age >= 40` |
| between 30 and 45 | age=between, ages 30–45 | kids=has_kids, age=between, ages 30–45 | `kids > 0 AND age >= 30 AND age <= 45` |
| thirtysomethings | age=decade, ages 30–39 | (none) | `TRUE` |
| seniors | age=seniors, ages 65– | state=NC, kids=no_kids, age=seniors, ages 65– | `state = 'NC' AND kids = 0 AND age >= 65` |
| young professionals | age=young, ages 18–34 | age=young, marital=single, ages 18–34 | `age >= 18 AND age <= 34 AND marital = 'single'` |
| new york no kids | state=NY, kids=no_kids | state=NY, kids=no_kids, age=young, marital=single, ages 18–34 | `state = 'NY' AND kids = 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |
| NYC parents | state=NY, kids=has_kids | state=NY, kids=has_kids, age=middle_aged, ages 40–59 | `state = 'NY' AND kids > 0 AND age >= 40 AND age <= 59` |
| child free in california | state=CA, kids=no_kids | state=CA, kids=no_kids, marital=single | `state = 'CA' AND kids = 0 AND marital = 'single'` |
| chicago clients without kids | state=IL, kids=no_kids | state=IL, kids=no_kids, age=young, ages 18–34 | `state = 'IL' AND kids = 0 AND age >= 18 AND age <= 34` |
| 50+ and married | age=older_than, marital=married, ages 50– | kids=has_kids, age=older_than, marital=married, ages 50– | `kids > 0 AND age >= 50 AND marital = 'married'` |
| widows in florida | state=FL, marital=widowed | state=FL, kids=no_kids, marital=widowed | `state = 'FL' AND kids = 0 AND marital = 'widowed'` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, marital=married | `state = 'CO' AND kids = 0 AND marital = 'married'` |
| with kids over 10 | kids=has_kids | kids=has_kids, age=older_than, ages 11– | `kids > 0 AND age >= 11` |
| recent signups | (none) | age=young, ages 18–34 | `age >= 18 AND age <= 34` |
| vip clients | (none) | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| clientes en texas sin hijos | state=TX, kids=no_kids | state=TX, kids=no_kids, marital=single | `state = 'TX' AND kids = 0 AND marital = 'single'` |
| solteros menores de 30 | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| sin niños en nuevo méxico | state=NM, kids=no_kids | state=NM, kids=no_kids, age=young, ages 18–34 | `state = 'NM' AND kids = 0 AND age >= 18 AND age <= 34` |
| jóvenes en miami | state=FL, age=young, ages 18–34 | state=FL, age=young, marital=single, ages 18–34 | `state = 'FL' AND age >= 18 AND age <= 34 AND marital = 'single'` |
| clients au texas sans enfants | state=TX, kids=no_kids | state=TX, kids=no_kids, marital=single | `state = 'TX' AND kids = 0 AND marital = 'single'` |
| célibataires moins de 30 ans | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| veufs | marital=widowed | kids=no_kids, marital=widowed | `kids = 0 AND marital = 'widowed'` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA | `state = 'MA'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| people | kids=has_kids |
| anyone in | state=WA, marital=single |
| contacts | kids=has_kids |
| leads | kids=has_kids |
| leads in | kids=no_kids |
| clients in washington | state=WA |
| people | kids=has_kids |
| the | kids=has_kids |
| the big | age=middle_aged |
| the | kids=has_kids |
| the bay | state=WA, age=middle_aged |
| socal | kids=has_kids, age=middle_aged |
| anyone in | state=WA, marital=single |
| anyone in the | marital=single |
| anyone in the lone | age=middle_aged, marital=single |
| anyone in the lone star | age=middle_aged, marital=single |
| clients with | kids=no_kids |
| child | kids=has_kids, age=young, marital=single |
| not | kids=no_kids |
| over | kids=no_kids |
| over | kids=no_kids |
| under | kids=no_kids |
| under | kids=no_kids |
| 40 | kids=has_kids, age=around |
| 40 and | kids=no_kids |
| between | state=WA, kids=has_kids |
| aged 35 | age=around |
| in | state=WA, kids=has_kids, marital=single |
| in their | kids=no_kids |
| in | state=WA, kids=has_kids, marital=single |
| in their | kids=no_kids |
| around | kids=has_kids |
| 40 | kids=has_kids |
| young | marital=single |
| older than | kids=no_kids |
| younger | kids=no_kids |
| new | marital=single |
| new york with kids | age=young |
| new york with kids over | marital=single |
| new | kids=has_kids, marital=single |
| new york no | age=middle_aged |
| child | kids=has_kids, age=young, marital=single |
| child free | marital=single |
| child free in | marital=single |
| 0 kids | marital=single |
| 0 kids under | marital=single |
| single | kids=no_kids |
| single moms | age=middle_aged |
| widowed over | kids=no_kids, age=seniors |
| clients in their | kids=no_kids, age=middle_aged |
| families | age=middle_aged |
| families in | state=NC |
| between | state=WA, kids=has_kids |
| between 30 and 45 | kids=has_kids |
| between 30 and 45 in | kids=no_kids |
| between 30 and 45 in new | state=NY, kids=has_kids |
| 50+ | kids=has_kids |
| 50+ and | kids=has_kids |
| young | marital=single |
| young professionals | marital=single |
| young professionals in | marital=single |
| single | kids=no_kids |
| single under | state=OR, kids=no_kids |
| single under 30 | kids=no_kids |
| single under 30 in | kids=no_kids |
| texas parents | age=middle_aged |
| divorced in | kids=no_kids |
| people | kids=has_kids |
| people over | state=WA |
| married with children in new | state=NM |
| widows | kids=no_kids, age=seniors |
| widows in | kids=no_kids, age=seniors |
| seniors | state=NC, kids=no_kids |
| seniors in | kids=no_kids, marital=single |
| single | kids=no_kids |
| single in | kids=no_kids |
| childless | marital=single |
| childless couples | marital=married |
| childless couples in | marital=married |
| moms | age=middle_aged |
| moms in | age=middle_aged, marital=single |
| moms in their | age=middle_aged, marital=single |
| married no | marital=divorced |
| clients in north | state=WA |
| clients in north carolina aged 25 | age=younger_than |
| divorced over | kids=no_kids |
| new | kids=has_kids, marital=single |
| young | marital=single |
| young parents | marital=single |
| young parents in | marital=single |
| colorado single | age=middle_aged |
| with kids | age=young, marital=single |
| with kids over | marital=single |
| clients with | kids=no_kids |
| recent | age=young |
| top | kids=has_kids, age=middle_aged |
| top accounts | age=middle_aged |
| vip | age=middle_aged |
| new | kids=has_kids, marital=single |
| clients with | kids=no_kids |
| clients with a | kids=no_kids |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| anyone who | marital=single |
| anyone who mentioned | kids=has_kids, age=middle_aged, marital=single |
| nueva | age=middle_aged |
| casados en | kids=no_kids |
| solteros | kids=no_kids, age=middle_aged |
| solteros menores | kids=no_kids |
| solteros menores de | kids=no_kids |
| jubilados | kids=no_kids, marital=widowed |
| jubilados en | marital=widowed |
| padres | age=middle_aged |
| personas | kids=has_kids |
| personas de | kids=no_kids, marital=single |
| personas de 40 | kids=no_kids |
| personas de 40 a | kids=has_kids |
| viudas | kids=no_kids |
| viudas en | kids=no_kids, age=seniors |
| en | kids=has_kids |
| en sus | kids=no_kids |
| sin | state=WA, marital=single |
| sin niños en | marital=single |
| sin niños en nuevo | age=young |
| mayores | kids=has_kids |
| mayores de | kids=has_kids |
| jóvenes | kids=has_kids, marital=single |
| jóvenes en | kids=has_kids, marital=single |
| madres | age=middle_aged, marital=divorced |
| madres solteras | age=middle_aged |
| madres solteras en nueva | state=NM, age=middle_aged |
| new | marital=single |
| new york avec enfants | age=young |
| new york avec enfants plus | age=young, marital=single |
| new york avec enfants plus de | marital=single |
| clients au | age=middle_aged |
| mariés en | kids=no_kids |
| célibataires | kids=no_kids |
| célibataires moins | kids=no_kids |
| célibataires moins de | kids=no_kids |
| célibataires moins de 30 | kids=no_kids |
| retraités | marital=widowed |
| retraités en | marital=widowed |
| entre 35 | kids=no_kids |
| entre 35 et | kids=no_kids |
| la | state=LA, kids=has_kids |
| familles | age=middle_aged |
| familles à | age=middle_aged |
| pas d'enfants en | marital=single |
| jeunes | kids=has_kids, marital=single |
| jeunes à | kids=has_kids, marital=single |
| mères | age=middle_aged |
| mères célibataires | age=middle_aged |
