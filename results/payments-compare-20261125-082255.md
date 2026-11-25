# Card transaction search: Gemma head, no examples vs Gemma head, trained vs Gemma 4 12B writing JSON vs Jev vs Laya vs Laya, trained

144 queries typed word by word (397 updates). Thresholds: Gemma head, no examples 0.9, Gemma head, trained 0.5, Jev 0.5, Laya 0.9, Laya, trained 0.5. Runs: `payments-head-20261124-204100.json`, `payments-head-trained-20261124-205829.json`, `payments-llm-20261124-210130.json`, `payments-jev-20261124-204244.json`, `payments-laya-20261124-215935.json`, `payments-laya-trained-20261125-082215.json`.

| | Gemma head, no examples | Gemma head, trained | Gemma 4 12B writing JSON | Jev | Laya | Laya, trained |
|---|---|---|---|---|---|---|
| Time per update, p50 / p95 | 436 / 508 ms | 424 / 433 ms | 1561 / 2035 ms | 137 / 211 ms | 75 / 175 ms | 75 / 78 ms |
| Finished queries with every filter right | 56.9% | 91.7% | 93.8% | 91.7% | 26.4% | 91.7% |
| category | 93.1% | 96.5% | 98.6% | 92.4% | 66.0% | 93.8% |
| channel | 79.9% | 98.6% | 100.0% | 100.0% | 90.3% | 98.6% |
| amount_range | 98.6% | 98.6% | 99.3% | 100.0% | 79.9% | 100.0% |
| date_value | 95.1% | 97.9% | 96.5% | 99.3% | 80.6% | 99.3% |
| state | 99.3% | 100.0% | 99.3% | 100.0% | 98.6% | 100.0% |
| fraud | 81.2% | 98.6% | 100.0% | 100.0% | 88.9% | 99.3% |
| Every filter right, en (102) | 56.9% | 94.1% | 95.1% | 93.1% | 26.5% | 93.1% |
| Every filter right, es (21) | 57.1% | 90.5% | 95.2% | 95.2% | 28.6% | 100.0% |
| Every filter right, fr (21) | 57.1% | 81.0% | 85.7% | 81.0% | 23.8% | 76.2% |
| Wrong filter while typing (of 253 half-typed) | 27.3% | 2.4% | 5.5% | 8.3% | 9.5% | 2.8% |

## Gemma head, no examples by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 34.0% | 68.8% |
| 0.3 | 34.0% | 68.8% |
| 0.5 | 36.8% | 64.8% |
| 0.6 | 40.3% | 55.3% |
| 0.7 | 47.2% | 49.0% |
| 0.8 | 54.2% | 37.5% |
| 0.9 | 56.9% | 27.3% |

## Gemma head, trained by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 91.7% | 2.4% |
| 0.3 | 91.7% | 2.4% |
| 0.5 | 91.7% | 2.4% |
| 0.6 | 91.0% | 2.4% |
| 0.7 | 87.5% | 1.6% |
| 0.8 | 84.7% | 1.2% |
| 0.9 | 81.9% | 0.4% |

## Jev by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 90.3% | 10.7% |
| 0.3 | 90.3% | 10.3% |
| 0.5 | 91.7% | 8.3% |
| 0.6 | 91.0% | 6.7% |
| 0.7 | 91.7% | 5.1% |
| 0.8 | 91.0% | 3.2% |
| 0.9 | 86.8% | 0.8% |

## Laya by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 0.0% | 99.6% |
| 0.3 | 1.4% | 98.8% |
| 0.5 | 4.2% | 88.9% |
| 0.6 | 16.0% | 75.1% |
| 0.7 | 21.5% | 51.8% |
| 0.8 | 29.2% | 21.3% |
| 0.9 | 26.4% | 9.5% |

## Laya, trained by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 91.7% | 3.6% |
| 0.3 | 91.7% | 2.8% |
| 0.5 | 91.7% | 2.8% |
| 0.6 | 91.7% | 2.8% |
| 0.7 | 91.7% | 2.4% |
| 0.8 | 91.7% | 2.4% |
| 0.9 | 90.3% | 2.0% |

## Gemma head, no examples: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| fuel purchases | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| uber rides | {'channel': None} | {'channel': 'in_person'} |
| petrol | {'channel': None} | {'channel': 'in_person'} |
| gas station visits | {'channel': None} | {'channel': 'in_person'} |
| groceries | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| supermarket spending | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| grocery store runs | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| restaurants | {'channel': None} | {'channel': 'in_person'} |
| dining out | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| coffee shops | {'channel': None} | {'channel': 'in_person'} |
| takeout | {'channel': None} | {'channel': 'in_person'} |
| pharmacy charges | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| haircuts | {'channel': None} | {'channel': 'in_person'} |
| salon visits | {'channel': None} | {'channel': 'in_person'} |
| hardware store | {'channel': None} | {'channel': 'in_person'} |
| vet bills | {'fraud': None} | {'fraud': 'legit'} |
| clothes shopping | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| apple store purchases | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| online purchases | {'category': None, 'fraud': None} | {'category': 'shopping', 'fraud': 'legit'} |
| online orders | {'category': None} | {'category': 'shopping'} |
| e-commerce transactions | {'category': None} | {'category': 'shopping'} |
| in-store purchases | {'fraud': None} | {'fraud': 'legit'} |
| card present transactions | {'fraud': None} | {'fraud': 'legit'} |
| paid in person | {'fraud': None} | {'fraud': 'legit'} |
| purchases above 500 dollars | {'fraud': None} | {'fraud': 'legit'} |
| less than 10 bucks | {'fraud': None} | {'fraud': 'legit'} |
| big purchases | {'amount_range': [500, None]} | {'amount_range': None} |
| transactions from july | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| gas in indiana | {'channel': None} | {'channel': 'in_person'} |
| gas over $50 | {'channel': None} | {'channel': 'in_person'} |
| online grocery orders | {'fraud': None} | {'fraud': 'legit'} |
| groceries last month | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| suspicious online purchases | {'category': None} | {'category': 'shopping'} |
| weekend dining | {'channel': None} | {'channel': 'in_person'} |
| fraudulent gas station charges | {'channel': None} | {'channel': 'in_person'} |
| pet store purchases under $30 | {'fraud': None} | {'fraud': 'legit'} |
| fraud in new york last week | {'fraud': 'fraud'} | {'fraud': None} |
| big travel expenses | {'amount_range': [500, None]} | {'amount_range': None} |
| legit grocery charges in may | {'date_value': '2026-05-01..2026-05-31'} | {'date_value': '2026-10-01..2026-10-31'} |
| coffee under 5 bucks | {'fraud': None} | {'fraud': 'legit'} |
| uber rides yesterday | {'channel': None} | {'channel': 'in_person'} |
| small online purchases on sundays | {'category': None, 'fraud': None} | {'category': 'shopping', 'fraud': 'legit'} |
| orders for 2 people over $50 | {'fraud': None} | {'fraud': 'legit'} |
| the last 3 purchases | {'date_value': None} | {'date_value': '2026-11-18..2026-11-24'} |
| gasolina | {'channel': None} | {'channel': 'in_person'} |
| compras en línea | {'category': None} | {'category': 'shopping'} |
| supermercado | {'channel': None, 'fraud': None} | {'channel': 'in_person', 'fraud': 'legit'} |
| compras de menos de 20 dólares | {'fraud': None} | {'fraud': 'legit'} |
| gimnasio | {'channel': None, 'date_value': None} | {'channel': 'in_person', 'date_value': '2026-11-18..2026-11-24'} |
| peluquería | {'channel': None} | {'channel': 'in_person'} |
| juguetes en marzo | {'date_value': '2026-03-01..2026-03-31'} | {'date_value': None} |
| entre 50 y 100 dólares | {'fraud': None} | {'fraud': 'legit'} |
| compras legítimas | {'category': None} | {'category': 'shopping'} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| achats en ligne | {'category': None} | {'category': 'shopping'} |
| courses le mois dernier | {'category': 'grocery'} | {'category': None} |
| vols de plus de 500 dollars | {'fraud': None} | {'fraud': 'fraud'} |
| achats de moins de 20 dollars | {'fraud': None} | {'fraud': 'legit'} |
| salle de sport | {'channel': None} | {'channel': 'in_person'} |
| coiffeur | {'channel': None, 'state': None} | {'channel': 'in_person', 'state': 'CO'} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |
| hôtels en juillet | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |

## Gemma head, no examples: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| fuel | channel=in_person |
| uber | channel=online |
| gas station | channel=in_person |
| supermarket | channel=in_person, fraud=legit |
| grocery | channel=in_person, fraud=legit |
| grocery store | channel=in_person, fraud=legit |
| dining | channel=in_person |
| hotel | channel=in_person |
| gym | channel=in_person |
| pharmacy | channel=in_person, fraud=legit |
| salon | channel=in_person |
| apple store | channel=in_person |
| e-commerce | category=shopping |
| paid | fraud=legit |
| purchases above 500 | fraud=legit |
| around | category=gas_transport, date_value=2026-11-18..2026-11-24 |
| more than | fraud=fraud |
| in | channel=in_person, fraud=fraud |
| this | fraud=fraud |
| on | fraud=fraud |
| on | fraud=fraud |
| purchases in | channel=in_person |
| gas in | channel=in_person |
| online grocery | fraud=legit |
| groceries | channel=in_person, fraud=legit |
| groceries last | channel=in_person, fraud=legit |
| restaurants | channel=in_person |
| fraudulent gas | channel=in_person |
| fraudulent gas station | channel=in_person |
| pet store | channel=in_person |
| pet store purchases | channel=in_person, fraud=legit |
| pet store purchases under | channel=in_person, fraud=legit |
| gym | channel=in_person |
| gym charges this | channel=in_person |
| in-store shopping | fraud=legit |
| in-store shopping in | fraud=legit |
| fraud in new | state=NM |
| fraud in new york last | category=travel, date_value=2026-10-01..2026-10-31 |
| legit grocery | channel=in_person |
| uber | channel=online |
| uber rides | channel=in_person |
| haircuts | channel=in_person |
| haircuts in | channel=in_person |
| customers in georgia with | category=travel |
| small online purchases | category=shopping, fraud=legit |
| small online purchases on | category=shopping, fraud=legit |
| orders for 2 people over | amount_range=[2, None] |
| all | fraud=fraud |
| the last | date_value=2026-11-18..2026-11-24 |
| the last 3 | date_value=2026-11-18..2026-11-24 |
| compras | category=shopping |
| compras en | category=shopping |
| restaurantes | channel=in_person |
| compras | category=shopping |
| compras de | category=shopping |
| compras de menos de | category=shopping |
| compras de menos de 20 | fraud=legit |
| la | state=LA |
| la semana | state=LA |
| compras | category=shopping |
| compras en | category=shopping |
| compras | category=shopping |
| achats | category=grocery |
| restaurants | channel=in_person |
| vols de | fraud=fraud |
| vols de plus de 500 | fraud=fraud |
| achats | category=grocery |
| achats de moins de | fraud=legit |
| achats de moins de 20 | fraud=legit |

## Gemma head, trained: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| grocery store runs | {'channel': None} | {'channel': 'in_person'} |
| flights over 500 | {'amount_range': [500, None]} | {'amount_range': None} |
| amazon orders | {'category': 'shopping', 'channel': 'online'} | {'category': None, 'channel': None} |
| in-store shopping in ohio | {'category': 'shopping'} | {'category': None} |
| fraud in new york last week | {'fraud': 'fraud'} | {'fraud': None} |
| online over $200 flagged as fraud | {'amount_range': [200, None]} | {'amount_range': None} |
| ayer | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': '2026-11-24..2026-11-24'} |
| la semana pasada | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': None} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| courses le mois dernier | {'category': 'grocery'} | {'category': None} |
| vols de plus de 500 dollars | {'category': 'travel', 'fraud': None} | {'category': None, 'fraud': 'fraud'} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |

## Gemma head, trained: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| around | category=gas_transport |
| in | channel=in_person |
| la | state=LA |
| la semana | state=LA |
| vols de | fraud=fraud |
| vols de plus | fraud=fraud |

## Gemma 4 12B writing JSON: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| about 40 dollars | {'amount_range': [36, 44]} | {'amount_range': [None, 40]} |
| past week | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': '2026-11-17..2026-11-24'} |
| amazon orders | {'category': 'shopping'} | {'category': None} |
| fraud in new york last week | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': '2026-11-17..2026-11-24'} |
| all transactions in 2026 | {'date_value': None} | {'date_value': '2026-01-01..2026-12-31'} |
| la semana pasada | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': '2026-11-17..2026-11-24'} |
| courses le mois dernier | {'category': 'grocery'} | {'category': None} |
| restaurants à new york | {'state': 'NY'} | {'state': None} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |

## Gemma 4 12B writing JSON: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| between $50 | amount_range=[50, None] |
| between $50 and | amount_range=[50, None] |
| about 40 | amount_range=[None, 40] |
| pet store purchases under | amount_range=[None, 20] |
| fraud in new york last | date_value=2026-11-17..2026-11-24 |
| coffee under | amount_range=[None, 20] |
| hotels between 100 | amount_range=[100, None] |
| hotels between 100 and | amount_range=[100, None] |
| entre 50 | amount_range=[50, None] |
| entre 50 y | amount_range=[50, None] |
| la semana | date_value=2026-11-17..2026-11-24 |
| courses le | state=LA |
| entre 50 | amount_range=[50, None] |
| entre 50 et | amount_range=[50, None] |

## Jev: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| online purchases | {'category': None} | {'category': 'shopping'} |
| online orders | {'category': None} | {'category': 'shopping'} |
| e-commerce transactions | {'category': None} | {'category': 'shopping'} |
| in-store purchases | {'category': None} | {'category': 'shopping'} |
| suspicious online purchases | {'category': None} | {'category': 'shopping'} |
| small online purchases on sundays | {'category': None} | {'category': 'shopping'} |
| orders for 2 people over $50 | {'category': None} | {'category': 'food_dining'} |
| compras en línea | {'category': None} | {'category': 'shopping'} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| achats en ligne | {'category': None} | {'category': 'shopping'} |
| courses le mois dernier | {'category': 'grocery'} | {'category': None} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |

## Jev: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| e-commerce | category=shopping |
| around | category=gas_transport |
| cardholders in | channel=in_person |
| small online | category=shopping |
| small online purchases | category=shopping |
| small online purchases on | category=shopping |
| orders for 2 people | category=food_dining |
| orders for 2 people over | category=food_dining |
| compras | category=shopping |
| compras en | category=shopping |
| compras | category=shopping |
| compras de | category=shopping |
| los | state=CA |
| los | state=CA |
| la | state=CA |
| compras | category=shopping |
| compras en | category=shopping |
| compras | category=shopping |
| achats | category=shopping |
| achats en | category=shopping |
| achats | category=shopping |

## Laya: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| uber rides | {'category': 'gas_transport'} | {'category': None} |
| petrol | {'category': 'gas_transport'} | {'category': None} |
| dining out | {'category': 'food_dining'} | {'category': None} |
| coffee shops | {'category': 'food_dining'} | {'category': None} |
| takeout | {'category': 'food_dining'} | {'category': None} |
| flights | {'category': 'travel'} | {'category': None} |
| hotel bookings | {'category': 'travel'} | {'category': None} |
| airfare | {'category': 'travel'} | {'category': None} |
| concerts | {'category': 'entertainment'} | {'category': None} |
| gym memberships | {'category': 'health_fitness'} | {'category': None} |
| pharmacy charges | {'category': 'health_fitness'} | {'category': None} |
| doctor visits | {'category': 'health_fitness'} | {'category': None} |
| haircuts | {'category': 'personal_care'} | {'category': None} |
| salon visits | {'category': 'personal_care'} | {'category': None} |
| beauty products | {'category': 'personal_care'} | {'category': None} |
| hardware store | {'category': 'home'} | {'category': None} |
| pet supplies | {'category': 'kids_pets'} | {'category': None} |
| vet bills | {'category': 'kids_pets'} | {'category': None} |
| toys | {'category': 'kids_pets'} | {'category': None} |
| childcare | {'category': 'kids_pets'} | {'category': None} |
| electronics | {'category': 'shopping'} | {'category': None} |
| apple store purchases | {'category': 'shopping'} | {'category': None} |
| online purchases | {'category': None, 'channel': 'online'} | {'category': 'shopping', 'channel': None} |
| online orders | {'channel': 'online'} | {'channel': None} |
| e-commerce transactions | {'channel': 'online'} | {'channel': None} |
| in-store purchases | {'channel': 'in_person'} | {'channel': None} |
| card present transactions | {'channel': 'in_person'} | {'channel': None} |
| paid in person | {'channel': 'in_person'} | {'channel': None} |
| flagged transactions | {'fraud': 'fraud'} | {'fraud': None} |
| fraudulent payments | {'fraud': 'fraud'} | {'fraud': None} |
| legit transactions only | {'fraud': 'legit'} | {'fraud': None} |
| non-fraud charges | {'fraud': 'legit'} | {'fraud': None} |
| genuine purchases | {'fraud': 'legit'} | {'fraud': None} |
| over $100 | {'amount_range': [100, None]} | {'amount_range': None} |
| charges under $20 | {'amount_range': [None, 20], 'fraud': None} | {'amount_range': None, 'fraud': 'fraud'} |
| purchases above 500 dollars | {'amount_range': [500, None]} | {'amount_range': None} |
| less than 10 bucks | {'amount_range': [None, 10], 'fraud': None} | {'amount_range': None, 'fraud': 'fraud'} |
| between $50 and $200 | {'amount_range': [50, 200]} | {'amount_range': None} |
| around $100 | {'amount_range': [90, 110]} | {'amount_range': None} |
| about 40 dollars | {'amount_range': [36, 44]} | {'amount_range': None} |
| more than 1k | {'amount_range': [1000, None]} | {'amount_range': None} |
| big purchases | {'amount_range': [500, None]} | {'amount_range': None} |
| small charges | {'amount_range': [None, 20]} | {'amount_range': None} |
| in march | {'date_value': '2026-03-01..2026-03-31'} | {'date_value': None} |
| transactions from july | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| last month | {'date_value': '2026-10-01..2026-10-31'} | {'date_value': None} |
| this month | {'date_value': '2026-11-01..2026-11-24'} | {'date_value': None} |
| on fridays | {'date_value': 'dow:4', 'state': None} | {'date_value': None, 'state': 'FL'} |
| saturday purchases | {'date_value': 'dow:5'} | {'date_value': None} |
| charges in california | {'category': None} | {'category': 'travel'} |
| spending in florida | {'category': None} | {'category': 'travel'} |
| gas over $50 | {'amount_range': [50, None]} | {'amount_range': None} |
| groceries last month | {'date_value': '2026-10-01..2026-10-31'} | {'date_value': None} |
| fraud over $1000 | {'amount_range': [1000, None]} | {'amount_range': None} |
| suspicious online purchases | {'channel': 'online'} | {'channel': None} |
| travel in july | {'category': 'travel', 'date_value': '2026-07-01..2026-07-31'} | {'category': None, 'date_value': None} |
| flights over 500 | {'category': 'travel', 'amount_range': [500, None]} | {'category': None, 'amount_range': None} |
| weekend dining | {'category': 'food_dining', 'date_value': 'dow:5,6'} | {'category': None, 'date_value': None} |
| amazon orders | {'channel': 'online'} | {'channel': None} |
| pet store purchases under $30 | {'amount_range': [None, 30]} | {'amount_range': None} |
| gym charges this month | {'category': 'health_fitness', 'date_value': '2026-11-01..2026-11-24'} | {'category': None, 'date_value': None} |
| in-store shopping in ohio | {'category': 'shopping', 'channel': 'in_person'} | {'category': None, 'channel': None} |
| fraud in new york last week | {'date_value': '2026-11-18..2026-11-24', 'fraud': 'fraud'} | {'date_value': None, 'fraud': None} |
| entertainment on saturdays | {'date_value': 'dow:5'} | {'date_value': None} |
| big travel expenses | {'category': 'travel', 'amount_range': [500, None]} | {'category': None, 'amount_range': None} |
| legit grocery charges in may | {'date_value': '2026-05-01..2026-05-31', 'fraud': 'legit'} | {'date_value': None, 'fraud': None} |
| coffee under 5 bucks | {'category': 'food_dining', 'amount_range': [None, 5]} | {'category': None, 'amount_range': None} |
| hotels between 100 and 300 | {'category': 'travel', 'amount_range': [100, 300]} | {'category': None, 'amount_range': None} |
| online over $200 flagged as fraud | {'channel': 'online', 'amount_range': [200, None], 'fraud': 'fraud'} | {'channel': None, 'amount_range': None, 'fraud': None} |
| furniture in florida over 1000 | {'category': 'home', 'amount_range': [1000, None]} | {'category': None, 'amount_range': None} |
| haircuts in april | {'category': 'personal_care', 'date_value': '2026-04-01..2026-04-30'} | {'category': None, 'date_value': None} |
| customers in georgia with fraud | {'fraud': 'fraud'} | {'fraud': None} |
| small online purchases on sundays | {'channel': 'online', 'amount_range': [None, 20], 'date_value': 'dow:6'} | {'channel': None, 'amount_range': None, 'date_value': None} |
| charges that may be fraud | {'fraud': 'fraud'} | {'fraud': None} |
| orders for 2 people over $50 | {'amount_range': [50, None]} | {'amount_range': None} |
| compras en línea | {'category': None} | {'category': 'shopping'} |
| vuelos de más de 500 dólares | {'category': 'travel', 'amount_range': [500, None]} | {'category': None, 'amount_range': None} |
| fraude el mes pasado | {'date_value': '2026-10-01..2026-10-31'} | {'date_value': None} |
| compras de menos de 20 dólares | {'category': None, 'amount_range': [None, 20], 'fraud': None} | {'category': 'shopping', 'amount_range': None, 'fraud': 'fraud'} |
| peluquería | {'category': 'personal_care'} | {'category': None} |
| juguetes en marzo | {'category': 'kids_pets', 'channel': None, 'date_value': '2026-03-01..2026-03-31'} | {'category': None, 'channel': 'online', 'date_value': None} |
| ropa en la tienda | {'channel': 'in_person'} | {'channel': None} |
| entre 50 y 100 dólares | {'amount_range': [50, 100], 'fraud': None} | {'amount_range': None, 'fraud': 'fraud'} |
| los fines de semana | {'date_value': 'dow:5,6'} | {'date_value': None} |
| los viernes | {'date_value': 'dow:4'} | {'date_value': None} |
| la semana pasada | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': None} |
| compras en florida este mes | {'category': None, 'date_value': '2026-11-01..2026-11-24'} | {'category': 'travel', 'date_value': None} |
| hoteles en julio | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| muebles de más de 1000 | {'amount_range': [1000, None]} | {'amount_range': None} |
| compras legítimas | {'category': None, 'fraud': 'legit'} | {'category': 'shopping', 'fraud': None} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| achats en ligne | {'category': None} | {'category': 'shopping'} |
| courses le mois dernier | {'category': 'grocery', 'date_value': '2026-10-01..2026-10-31'} | {'category': None, 'date_value': None} |
| vols de plus de 500 dollars | {'category': 'travel', 'amount_range': [500, None]} | {'category': None, 'amount_range': None} |
| achats de moins de 20 dollars | {'category': None, 'amount_range': [None, 20]} | {'category': 'shopping', 'amount_range': None} |
| salle de sport | {'category': 'health_fitness'} | {'category': None} |
| animalerie | {'category': 'kids_pets'} | {'category': None} |
| vêtements en magasin | {'channel': 'in_person'} | {'channel': None} |
| entre 50 et 100 dollars | {'amount_range': [50, 100]} | {'amount_range': None} |
| le week-end | {'date_value': 'dow:5,6'} | {'date_value': None} |
| le vendredi | {'date_value': 'dow:4'} | {'date_value': None} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |
| pas de fraude | {'fraud': 'legit'} | {'fraud': 'fraud'} |
| hôtels en juillet | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| dépenses au texas ce mois-ci | {'date_value': '2026-11-01..2026-11-24', 'state': 'TX'} | {'date_value': None, 'state': None} |
| environ 100 dollars | {'amount_range': [90, 110]} | {'amount_range': None} |

## Laya: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| customers from new | state=NH |
| fraud in new | state=NH |
| customers in georgia | category=travel |
| compras | category=shopping |
| compras en | category=shopping |
| compras | category=shopping |
| compras de | category=shopping |
| compras de menos | category=shopping |
| compras de menos de | category=shopping |
| compras de menos de 20 | category=shopping |
| compras | category=shopping |
| compras en | category=shopping |
| compras en florida este | category=travel |
| muebles de más de | fraud=fraud |
| compras | category=shopping |
| achats en | category=shopping |
| restaurants à new | state=NH |
| achats de | category=shopping |
| achats de moins | category=shopping |
| achats de moins de | category=shopping |
| achats de moins de 20 | category=shopping |
| dépenses au | state=UT |
| dépenses au texas | state=UT |
| dépenses au texas ce | state=UT |

## Laya, trained: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| airfare | {'category': 'travel'} | {'category': None} |
| salon visits | {'category': 'personal_care'} | {'category': None} |
| apple store purchases | {'category': 'shopping'} | {'category': None} |
| card present transactions | {'channel': 'in_person'} | {'channel': None} |
| on fridays | {'date_value': 'dow:4'} | {'date_value': 'dow:5,6'} |
| amazon orders | {'category': 'shopping', 'channel': 'online'} | {'category': None, 'channel': None} |
| in-store shopping in ohio | {'category': 'shopping'} | {'category': None} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| vols de plus de 500 dollars | {'category': 'travel'} | {'category': None} |
| salle de sport | {'category': 'health_fitness'} | {'category': 'entertainment'} |
| coiffeur | {'category': 'personal_care'} | {'category': None} |
| pas de fraude | {'fraud': 'legit'} | {'fraud': 'fraud'} |

## Laya, trained: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| the last | date_value=2026-11-18..2026-11-24 |
| the last 3 | date_value=2026-11-18..2026-11-24 |
| fraude el mes | date_value=2026-11-01..2026-11-24 |
| la semana | date_value=dow:5,6 |
| courses | category=food_dining |
| courses le | category=food_dining |
| courses le mois | date_value=2026-11-01..2026-11-24 |
