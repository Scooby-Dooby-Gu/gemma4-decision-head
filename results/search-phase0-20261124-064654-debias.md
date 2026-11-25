# Search POC phase 0: order-averaged, with the blank-input prior

Model `mlx-community/gemma-4-e4b-it-4bit`, 134 queries typed word by word = 439 updates, 10 forward passes per update. Model load 2.6 s; first query (prompts, priors, prefix caches) 3.0 s.

## Latency per update (all filter questions for one prefix)

| p50 | p95 | mean | max |
|---|---|---|---|
| 638 ms | 655 ms | 630 ms | 661 ms |

## The head on its own (top option, no threshold)

| Question | Finished query | Partly typed |
|---|---|---|
| state_east | 53.7% | 20.3% |
| state_west | 88.8% | 61.6% |
| kids | 85.8% | 48.5% |
| age | 70.9% | 37.7% |
| marital | 87.3% | 65.2% |

## Applied filters by confidence threshold

A filter is applied when the head picks a value, doesn't abstain (top two within 0.10) and its confidence is at least the threshold. *Wrong on the way* counts partly typed prefixes with a filter applied that is neither right yet nor where the query ends up.

| Threshold | All filters right | state | kids | age kind | age range | marital | Wrong on the way (prefixes) | Applied early but right | Missed while typing |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 63.4% | 97.0% | 88.8% | 80.6% | 80.6% | 92.5% | 163 of 305 (53.4%) | 43 | 3 of 201 |
| 0.7 | 77.6% | 97.0% | 91.8% | 90.3% | 90.3% | 95.5% | 96 of 305 (31.5%) | 34 | 4 of 201 |
| 0.9 | 81.3% | 97.0% | 92.5% | 93.3% | 93.3% | 97.8% | 52 of 305 (17.0%) | 29 | 7 of 201 |

All filters right, by language:

| Threshold | en | es | fr |
|---|---|---|---|
| 0.5 | 63.8% (105) | 66.7% (15) | 57.1% (14) |
| 0.7 | 79.0% (105) | 66.7% (15) | 78.6% (14) |
| 0.9 | 84.8% (105) | 60.0% (15) | 78.6% (14) |

## Finished queries with a wrong filter (threshold 0.5)

| Query | Expected | Got | SQL |
|---|---|---|---|
| clients in washington dc | state=DC | state=WA | `state = 'WA'` |
| the big apple | state=NY | state=MA | `state = 'MA'` |
| socal clients | state=CA | (none) | `TRUE` |
| anyone in the lone star state | state=TX | state=TX, marital=single | `state = 'TX' AND marital = 'single'` |
| nola | state=LA | state=LA, kids=has_kids | `state = 'LA' AND kids > 0` |
| with kids | kids=has_kids | kids=has_kids, age=young, ages 18–34 | `kids > 0 AND age >= 18 AND age <= 34` |
| parents | kids=has_kids | kids=has_kids, age=middle_aged, ages 40–59 | `kids > 0 AND age >= 40 AND age <= 59` |
| moms | kids=has_kids | kids=has_kids, age=middle_aged, ages 40–59 | `kids > 0 AND age >= 40 AND age <= 59` |
| child free | kids=no_kids | kids=no_kids, age=young, marital=single, ages 18–34 | `kids = 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |
| 0 kids | kids=no_kids | kids=no_kids, age=young, ages 18–34 | `kids = 0 AND age >= 18 AND age <= 34` |
| no kids | kids=no_kids | kids=no_kids, age=young, ages 18–34 | `kids = 0 AND age >= 18 AND age <= 34` |
| childless | kids=no_kids | kids=no_kids, age=young, marital=single, ages 18–34 | `kids = 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |
| without children | kids=no_kids | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| single | marital=single | kids=no_kids, marital=single | `kids = 0 AND marital = 'single'` |
| widowed | marital=widowed | age=seniors, marital=widowed, ages 65– | `age >= 65 AND marital = 'widowed'` |
| over 40 | age=older_than, ages 41– | kids=has_kids, age=older_than, ages 41– | `kids > 0 AND age >= 41` |
| under 30 | age=younger_than, ages –29 | kids=no_kids, age=younger_than, ages –29 | `kids = 0 AND age <= 29` |
| under 25 | age=younger_than, ages –24 | kids=no_kids, age=younger_than, marital=single, ages –24 | `kids = 0 AND age <= 24 AND marital = 'single'` |
| 50+ | age=older_than, ages 50– | kids=has_kids, age=older_than, ages 50– | `kids > 0 AND age >= 50` |
| 40 and up | age=older_than, ages 40– | kids=has_kids, age=older_than, ages 40– | `kids > 0 AND age >= 40` |
| between 30 and 45 | age=between, ages 30–45 | kids=has_kids, age=between, ages 30–45 | `kids > 0 AND age >= 30 AND age <= 45` |
| thirtysomethings | age=decade, ages 30–39 | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| around 50 | age=around, ages 48–52 | kids=has_kids, age=around, ages 48–52 | `kids > 0 AND age >= 48 AND age <= 52` |
| young professionals | age=young, ages 18–34 | age=young, marital=single, ages 18–34 | `age >= 18 AND age <= 34 AND marital = 'single'` |
| new york no kids | state=NY, kids=no_kids | state=NY, kids=no_kids, age=young, ages 18–34 | `state = 'NY' AND kids = 0 AND age >= 18 AND age <= 34` |
| NYC parents | state=NY, kids=has_kids | state=NY, kids=has_kids, age=middle_aged, ages 40–59 | `state = 'NY' AND kids > 0 AND age >= 40 AND age <= 59` |
| child free in california | state=CA, kids=no_kids | state=CA, kids=no_kids, age=young, marital=single, ages 18–34 | `state = 'CA' AND kids = 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |
| 0 kids under 35 | kids=no_kids, age=younger_than, ages –34 | kids=no_kids, age=younger_than, marital=single, ages –34 | `kids = 0 AND age <= 34 AND marital = 'single'` |
| chicago clients without kids | state=IL, kids=no_kids | state=IL, kids=no_kids, age=young, ages 18–34 | `state = 'IL' AND kids = 0 AND age >= 18 AND age <= 34` |
| 50+ and married | age=older_than, marital=married, ages 50– | kids=has_kids, age=older_than, marital=married, ages 50– | `kids > 0 AND age >= 50 AND marital = 'married'` |
| widows in florida | state=FL, marital=widowed | state=FL, kids=no_kids, marital=widowed | `state = 'FL' AND kids = 0 AND marital = 'widowed'` |
| childless couples in denver | state=CO, kids=no_kids | state=CO, kids=no_kids, age=young, marital=married, ages 18–34 | `state = 'CO' AND kids = 0 AND age >= 18 AND age <= 34 AND marital = 'married'` |
| parents of teenagers | kids=has_kids | kids=has_kids, age=young, ages 18–34 | `kids > 0 AND age >= 18 AND age <= 34` |
| with kids over 10 | kids=has_kids | kids=has_kids, age=older_than, ages 11– | `kids > 0 AND age >= 11` |
| clients with 3 kids | kids=has_kids | kids=has_kids, age=middle_aged, ages 40–59 | `kids > 0 AND age >= 40 AND age <= 59` |
| recent signups | (none) | age=young, ages 18–34 | `age >= 18 AND age <= 34` |
| vip clients | (none) | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| clients with a georgia phone number | (none) | state=GA | `state = 'GA'` |
| solteros menores de 30 | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| familias en chicago | state=IL, kids=has_kids | state=IL | `state = 'IL'` |
| en sus cuarentas | age=decade, ages 40–49 | age=middle_aged, ages 40–59 | `age >= 40 AND age <= 59` |
| sin niños en nuevo méxico | state=NM, kids=no_kids | state=NM, kids=no_kids, age=young, ages 18–34 | `state = 'NM' AND kids = 0 AND age >= 18 AND age <= 34` |
| jóvenes en miami | state=FL, age=young, ages 18–34 | state=FL, age=young, marital=single, ages 18–34 | `state = 'FL' AND age >= 18 AND age <= 34 AND marital = 'single'` |
| clients au texas sans enfants | state=TX, kids=no_kids | state=TX, kids=no_kids, age=young, ages 18–34 | `state = 'TX' AND kids = 0 AND age >= 18 AND age <= 34` |
| célibataires moins de 30 ans | age=younger_than, marital=single, ages –29 | kids=no_kids, age=younger_than, marital=single, ages –29 | `kids = 0 AND age <= 29 AND marital = 'single'` |
| veufs | marital=widowed | kids=has_kids, age=seniors, marital=widowed, ages 65– | `kids > 0 AND age >= 65 AND marital = 'widowed'` |
| la quarantaine à boston | state=MA, age=decade, ages 40–49 | state=MA | `state = 'MA'` |
| pas d'enfants en géorgie | state=GA, kids=no_kids | state=GA, kids=no_kids, age=young, ages 18–34 | `state = 'GA' AND kids = 0 AND age >= 18 AND age <= 34` |
| mères célibataires à miami | state=FL, kids=has_kids, marital=single | state=FL, kids=has_kids, age=young, marital=single, ages 18–34 | `state = 'FL' AND kids > 0 AND age >= 18 AND age <= 34 AND marital = 'single'` |

## Wrong filters while typing (threshold 0.5)

| Typed so far | Wrong filter |
|---|---|
| clients | kids=has_kids |
| clients in | kids=has_kids |
| people | kids=has_kids |
| people in | kids=has_kids |
| anyone in | marital=single |
| anyone in new | kids=has_kids |
| contacts | kids=has_kids |
| contacts in west | state=WA |
| clients | kids=has_kids |
| leads | kids=has_kids |
| leads in | kids=has_kids |
| clients | kids=has_kids |
| clients in | kids=has_kids |
| clients in washington | state=WA |
| people | kids=has_kids |
| people in | kids=has_kids |
| the | kids=has_kids |
| clients | kids=has_kids |
| clients in | kids=has_kids |
| the | kids=has_kids |
| the bay | state=WA |
| socal | kids=has_kids |
| las | kids=has_kids |
| anyone in | marital=single |
| anyone in the lone | marital=single |
| anyone in the lone star | marital=single |
| child | kids=has_kids, age=young, marital=single |
| 0 | kids=has_kids |
| over | kids=has_kids |
| over | kids=has_kids |
| 40 | kids=has_kids |
| 40 and | kids=has_kids |
| between | kids=has_kids |
| between 30 | kids=has_kids |
| aged 35 | age=around |
| aged 35 to | kids=no_kids |
| in | kids=has_kids |
| in | kids=has_kids |
| around | kids=has_kids |
| 40 | kids=has_kids |
| young | marital=single |
| middle | kids=has_kids |
| older than | kids=no_kids |
| younger | kids=has_kids |
| younger than | kids=no_kids |
| new york with kids | age=young |
| new | kids=has_kids |
| married couples | kids=has_kids |
| married couples in | kids=has_kids |
| child | kids=has_kids, age=young, marital=single |
| child free | age=young, marital=single |
| child free in | age=young, marital=single |
| 0 | kids=has_kids |
| 0 kids | age=young |
| 0 kids under | age=young, marital=single |
| single | kids=no_kids |
| single moms in | age=young |
| widowed | age=seniors |
| widowed over | kids=no_kids, age=seniors |
| clients | kids=has_kids |
| clients in | kids=has_kids |
| families | age=middle_aged |
| between | kids=has_kids |
| between 30 | kids=has_kids |
| between 30 and 45 | kids=has_kids |
| between 30 and 45 in | kids=has_kids |
| between 30 and 45 in new | kids=has_kids |
| 50+ | kids=has_kids |
| 50+ and | kids=has_kids |
| young | marital=single |
| young professionals | marital=single |
| young professionals in | marital=single |
| single | kids=no_kids |
| single under | kids=no_kids |
| single under 30 | kids=no_kids |
| single under 30 in | kids=no_kids |
| texas parents | age=middle_aged |
| divorced in | kids=no_kids |
| people | kids=has_kids |
| people over | kids=has_kids |
| people over 40 with no | marital=single |
| married with children in new | state=NM |
| widows | kids=no_kids, age=seniors |
| widows in | kids=no_kids, age=seniors |
| seniors in | kids=no_kids |
| single | kids=no_kids |
| single in | kids=no_kids |
| childless | age=young, marital=single |
| childless couples | age=young, marital=married |
| childless couples in | age=young, marital=married |
| clients | kids=has_kids |
| clients in | kids=has_kids |
| moms | age=middle_aged |
| moms in | marital=single |
| moms in their | marital=single |
| clients | kids=has_kids |
| clients in | kids=has_kids |
| clients in north | state=MT |
| clients in north carolina aged 25 | age=younger_than |
| parents | age=middle_aged |
| new | kids=has_kids |
| young | marital=single |
| young parents | marital=single |
| young parents in | marital=single |
| parents | age=middle_aged |
| with kids | age=young |
| recent | age=young |
| clients | kids=has_kids |
| clients who | kids=has_kids |
| top | kids=has_kids |
| vip | age=middle_aged |
| new | kids=has_kids |
| clients | kids=has_kids |
| clients with | kids=has_kids |
| clients with a | kids=has_kids |
| clients with a georgia | state=GA |
| clients with a georgia phone | state=GA |
| anyone who | age=young |
| anyone who mentioned | kids=has_kids, marital=single |
| nueva york con hijos | age=young |
| solteros | kids=no_kids |
| solteros menores | kids=no_kids |
| solteros menores de | kids=no_kids |
| jubilados en | marital=divorced |
| padres | age=middle_aged |
| personas | kids=has_kids |
| personas de 40 | kids=no_kids, age=middle_aged |
| personas de 40 a | kids=has_kids |
| viudas | age=seniors |
| viudas en | kids=no_kids, age=seniors |
| en | kids=has_kids |
| en sus | kids=has_kids |
| sin | marital=single |
| sin niños | kids=has_kids, age=young |
| sin niños en | age=young |
| sin niños en nuevo | kids=has_kids, age=young |
| mayores | kids=has_kids |
| mayores de | kids=has_kids |
| jóvenes | kids=has_kids, marital=single |
| jóvenes en | kids=has_kids, marital=single |
| madres | marital=divorced |
| madres solteras en nueva | state=NM |
| new york avec enfants | age=young |
| new york avec enfants plus | age=young |
| new york avec enfants plus de | marital=single |
| clients | kids=has_kids |
| célibataires | kids=no_kids, age=young |
| célibataires moins | kids=no_kids, age=young |
| célibataires moins de | kids=no_kids |
| célibataires moins de 30 | kids=no_kids |
| parents | age=middle_aged |
| entre | kids=has_kids |
| entre 35 | kids=has_kids |
| entre 35 et | kids=no_kids |
| la | state=LA, kids=has_kids |
| la quarantaine | state=LA, age=middle_aged |
| la quarantaine à | state=LA |
| pas | kids=has_kids |
| pas d'enfants | age=young |
| pas d'enfants en | age=young, marital=single |
| jeunes | kids=has_kids, marital=single |
| jeunes à | kids=has_kids, marital=single |
| mères | age=seniors |
