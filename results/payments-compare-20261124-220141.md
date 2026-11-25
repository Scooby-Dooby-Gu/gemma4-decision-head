# Card transaction search: Jev vs Laya vs Laya, English fallback vs Gemma head, trained

144 queries typed word by word (397 updates). Thresholds: Jev 0.5, Laya 0.9, Laya, English fallback 0.7, Gemma head, trained 0.5. Runs: `payments-jev-20261124-204244.json`, `payments-laya-20261124-215935.json`, `payments-laya-en-20261124-220012.json`, `payments-head-trained-20261124-205829.json`.

| | Jev | Laya | Laya, English fallback | Gemma head, trained |
|---|---|---|---|---|
| Time per update, p50 / p95 | 137 / 211 ms | 75 / 175 ms | 166 / 174 ms | 424 / 433 ms |
| Finished queries with every filter right | 91.7% | 26.4% | 23.6% | 91.7% |
| category | 92.4% | 66.0% | 64.6% | 96.5% |
| channel | 100.0% | 90.3% | 93.8% | 98.6% |
| amount_range | 100.0% | 79.9% | 90.3% | 98.6% |
| date_value | 99.3% | 80.6% | 78.5% | 97.9% |
| state | 100.0% | 98.6% | 99.3% | 100.0% |
| fraud | 100.0% | 88.9% | 81.9% | 98.6% |
| Every filter right, en (102) | 93.1% | 26.5% | 27.5% | 94.1% |
| Every filter right, es (21) | 95.2% | 28.6% | 9.5% | 90.5% |
| Every filter right, fr (21) | 81.0% | 23.8% | 19.0% | 81.0% |
| Wrong filter while typing (of 253 half-typed) | 8.3% | 9.5% | 8.7% | 2.4% |

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

## Laya, English fallback by threshold

| Threshold | Every filter right | Wrong while typing |
|---|---|---|
| 0.0 | 0.0% | 99.6% |
| 0.3 | 0.7% | 99.6% |
| 0.5 | 2.8% | 88.9% |
| 0.6 | 25.7% | 13.8% |
| 0.7 | 23.6% | 8.7% |
| 0.8 | 18.8% | 4.7% |
| 0.9 | 9.7% | 2.8% |

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

## Laya, English fallback: finished queries with a wrong filter

| Query | Expected | Got |
|---|---|---|
| uber rides | {'category': 'gas_transport'} | {'category': None} |
| restaurants | {'category': 'food_dining'} | {'category': None} |
| dining out | {'category': 'food_dining'} | {'category': None} |
| coffee shops | {'category': 'food_dining'} | {'category': None} |
| takeout | {'category': 'food_dining'} | {'category': None} |
| flights | {'category': 'travel'} | {'category': None} |
| hotel bookings | {'category': 'travel'} | {'category': None} |
| airfare | {'category': 'travel'} | {'category': None} |
| movie tickets | {'category': 'entertainment'} | {'category': None} |
| concerts | {'category': 'entertainment'} | {'category': None} |
| pharmacy charges | {'category': 'health_fitness'} | {'category': None} |
| doctor visits | {'category': 'health_fitness'} | {'category': None} |
| haircuts | {'category': 'personal_care'} | {'category': None} |
| beauty products | {'category': 'personal_care'} | {'category': None} |
| furniture | {'category': 'home'} | {'category': None} |
| home improvement | {'category': 'home'} | {'category': None} |
| hardware store | {'category': 'home'} | {'category': None} |
| pet supplies | {'category': 'kids_pets'} | {'category': None} |
| vet bills | {'category': 'kids_pets'} | {'category': None} |
| toys | {'category': 'kids_pets'} | {'category': None} |
| childcare | {'category': 'kids_pets'} | {'category': None} |
| electronics | {'category': 'shopping'} | {'category': None} |
| apple store purchases | {'category': 'shopping'} | {'category': None} |
| e-commerce transactions | {'channel': 'online'} | {'channel': None} |
| in-store purchases | {'channel': 'in_person'} | {'channel': None} |
| card present transactions | {'channel': 'in_person'} | {'channel': None} |
| paid in person | {'channel': 'in_person'} | {'channel': None} |
| fraud | {'fraud': 'fraud'} | {'fraud': None} |
| suspicious charges | {'fraud': 'fraud'} | {'fraud': None} |
| flagged transactions | {'fraud': 'fraud'} | {'fraud': None} |
| fraudulent payments | {'fraud': 'fraud'} | {'fraud': None} |
| legit transactions only | {'fraud': 'legit'} | {'fraud': None} |
| non-fraud charges | {'fraud': 'legit'} | {'fraud': None} |
| genuine purchases | {'fraud': 'legit'} | {'fraud': None} |
| less than 10 bucks | {'amount_range': [None, 10]} | {'amount_range': None} |
| between $50 and $200 | {'amount_range': [50, 200]} | {'amount_range': None} |
| about 40 dollars | {'amount_range': [36, 44]} | {'amount_range': None} |
| more than 1k | {'amount_range': [1000, None]} | {'amount_range': [500, None]} |
| big purchases | {'amount_range': [500, None]} | {'amount_range': None} |
| small charges | {'amount_range': [None, 20]} | {'amount_range': None} |
| in march | {'date_value': '2026-03-01..2026-03-31'} | {'date_value': None} |
| transactions from july | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| last month | {'date_value': '2026-10-01..2026-10-31'} | {'date_value': None} |
| this month | {'date_value': '2026-11-01..2026-11-24'} | {'date_value': None} |
| yesterday | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |
| past week | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': None} |
| on fridays | {'date_value': 'dow:4'} | {'date_value': None} |
| on weekends | {'date_value': 'dow:5,6'} | {'date_value': None} |
| saturday purchases | {'date_value': 'dow:5'} | {'date_value': None} |
| groceries last month | {'date_value': '2026-10-01..2026-10-31'} | {'date_value': None} |
| restaurants in texas | {'category': 'food_dining'} | {'category': None} |
| fraud over $1000 | {'fraud': 'fraud'} | {'fraud': None} |
| suspicious online purchases | {'fraud': 'fraud'} | {'fraud': None} |
| travel in july | {'category': 'travel', 'date_value': '2026-07-01..2026-07-31'} | {'category': None, 'date_value': None} |
| flights over 500 | {'category': 'travel'} | {'category': None} |
| weekend dining | {'category': 'food_dining', 'date_value': 'dow:5,6'} | {'category': None, 'date_value': None} |
| fraudulent gas station charges | {'fraud': 'fraud'} | {'fraud': None} |
| amazon orders | {'category': 'shopping', 'channel': 'online'} | {'category': None, 'channel': None} |
| gym charges this month | {'category': 'health_fitness', 'date_value': '2026-11-01..2026-11-24'} | {'category': None, 'date_value': None} |
| in-store shopping in ohio | {'category': 'shopping', 'channel': 'in_person'} | {'category': None, 'channel': None} |
| fraud in new york last week | {'date_value': '2026-11-18..2026-11-24', 'fraud': 'fraud'} | {'date_value': None, 'fraud': None} |
| entertainment on saturdays | {'date_value': 'dow:5'} | {'date_value': None} |
| big travel expenses | {'amount_range': [500, None]} | {'amount_range': None} |
| legit grocery charges in may | {'date_value': '2026-05-01..2026-05-31'} | {'date_value': None} |
| coffee under 5 bucks | {'category': 'food_dining'} | {'category': None} |
| hotels between 100 and 300 | {'amount_range': [100, 300]} | {'amount_range': None} |
| uber rides yesterday | {'category': 'gas_transport'} | {'category': None} |
| online over $200 flagged as fraud | {'amount_range': [200, None], 'fraud': 'fraud'} | {'amount_range': None, 'fraud': None} |
| furniture in florida over 1000 | {'category': 'home'} | {'category': None} |
| haircuts in april | {'category': 'personal_care', 'date_value': '2026-04-01..2026-04-30'} | {'category': None, 'date_value': None} |
| customers in georgia with fraud | {'fraud': 'fraud'} | {'fraud': None} |
| small online purchases on sundays | {'date_value': 'dow:6'} | {'date_value': 'dow:5,6'} |
| charges that may be fraud | {'fraud': 'fraud'} | {'fraud': None} |
| orders for 2 people over $50 | {'category': None} | {'category': 'grocery'} |
| gasolina | {'category': 'gas_transport'} | {'category': None} |
| compras en línea | {'category': None, 'fraud': None} | {'category': 'shopping', 'fraud': 'fraud'} |
| supermercado | {'category': 'grocery'} | {'category': None} |
| vuelos de más de 500 dólares | {'amount_range': [500, None], 'fraud': None} | {'amount_range': None, 'fraud': 'fraud'} |
| fraude el mes pasado | {'date_value': '2026-10-01..2026-10-31', 'fraud': 'fraud'} | {'date_value': None, 'fraud': None} |
| cargos sospechosos | {'fraud': 'fraud'} | {'fraud': None} |
| compras de menos de 20 dólares | {'category': None, 'amount_range': [None, 20], 'fraud': None} | {'category': 'shopping', 'amount_range': None, 'fraud': 'fraud'} |
| gimnasio | {'category': 'health_fitness'} | {'category': None} |
| peluquería | {'category': 'personal_care'} | {'category': None} |
| juguetes en marzo | {'category': 'kids_pets', 'date_value': '2026-03-01..2026-03-31'} | {'category': None, 'date_value': None} |
| ropa en la tienda | {'category': 'shopping', 'channel': 'in_person'} | {'category': None, 'channel': None} |
| entre 50 y 100 dólares | {'channel': None, 'amount_range': [50, 100], 'fraud': None} | {'channel': 'online', 'amount_range': None, 'fraud': 'fraud'} |
| los viernes | {'date_value': 'dow:4'} | {'date_value': None} |
| ayer | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |
| la semana pasada | {'date_value': '2026-11-18..2026-11-24'} | {'date_value': None} |
| compras en florida este mes | {'date_value': '2026-11-01..2026-11-24'} | {'date_value': None} |
| hoteles en julio | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| muebles de más de 1000 | {'amount_range': [1000, None], 'fraud': None} | {'amount_range': None, 'fraud': 'fraud'} |
| compras legítimas | {'category': None, 'fraud': 'legit'} | {'category': 'shopping', 'fraud': None} |
| essence | {'category': 'gas_transport'} | {'category': None} |
| courses le mois dernier | {'category': 'grocery', 'date_value': '2026-10-01..2026-10-31', 'fraud': None} | {'category': None, 'date_value': None, 'fraud': 'fraud'} |
| vols de plus de 500 dollars | {'category': 'travel'} | {'category': None} |
| fraude | {'fraud': 'fraud'} | {'fraud': None} |
| transactions suspectes en ligne | {'fraud': 'fraud'} | {'fraud': None} |
| salle de sport | {'category': 'health_fitness'} | {'category': None} |
| coiffeur | {'category': 'personal_care'} | {'category': None} |
| animalerie | {'category': 'kids_pets'} | {'category': None} |
| vêtements en magasin | {'channel': 'in_person'} | {'channel': None} |
| le week-end | {'date_value': 'dow:5,6'} | {'date_value': None} |
| le vendredi | {'date_value': 'dow:4'} | {'date_value': None} |
| hier | {'date_value': '2026-11-23..2026-11-23'} | {'date_value': None} |
| pas de fraude | {'fraud': 'legit'} | {'fraud': None} |
| hôtels en juillet | {'date_value': '2026-07-01..2026-07-31'} | {'date_value': None} |
| meubles en californie | {'category': 'home', 'state': 'CA'} | {'category': None, 'state': None} |
| dépenses au texas ce mois-ci | {'date_value': '2026-11-01..2026-11-24'} | {'date_value': None} |
| environ 100 dollars | {'amount_range': [90, 110]} | {'amount_range': None} |

## Laya, English fallback: wrong filters while typing

| Typed so far | Wrong filter |
|---|---|
| customers from new | state=NM |
| gym charges | category=gas_transport |
| gym charges this | category=gas_transport |
| small online purchases on | category=shopping |
| charges that may | category=gas_transport |
| charges that may be | category=gas_transport |
| orders for 2 people over | category=grocery |
| vuelos de más | fraud=fraud |
| vuelos de más de | fraud=fraud |
| cargos | category=grocery |
| muebles de más | fraud=fraud |
| muebles de más de | fraud=fraud |
| restaurants à new | state=NH |
| vols | category=gas_transport |
| vols de | category=gas_transport, state=DE |
| vêtements | fraud=fraud |
| vêtements en | fraud=fraud |
| hôtels | fraud=fraud |
| dépenses | fraud=fraud |
| dépenses au | state=UT, fraud=fraud |
| dépenses au texas | category=travel, state=UT |
| dépenses au texas ce | state=UT |

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
