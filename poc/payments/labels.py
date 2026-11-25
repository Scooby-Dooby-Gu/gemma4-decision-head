"""Generated training examples for the payment filter heads, kept apart from the test queries.

    python -m poc.payments.labels        # writes train_cases.json

The same method as poc/search/labels.py: short phrases per option in English, Spanish and French,
combined with filler words into queries, plus about half of each query's word-by-word prefixes.
A prefix that stops partway through a phrase leaves that phrase's question unlabelled ("skip").
No training text equals a test query or prefix, and no phrase the test set labels is used,
except a state's own name.
"""

import json
import random
from pathlib import Path

from poc.search.crm import STATES
from poc.search.filters import EAST, NONE, NOT_MENTIONED
from poc.search.labels import STATE_PREP, state_phrases

from .bench import parse_queries, prefixes
from .filters import CATEGORIES

HERE = Path(__file__).resolve().parent
OUT = HERE / "train_cases.json"
LANGS = ("en", "es", "fr")

CATEGORY = {
    "gas_transport": {"en": ["gas fill-ups", "fuel stops", "gas stations", "transit fares", "lyft rides", "taxi rides",
                             "parking and tolls", "filling up the tank", "bus passes", "subway fares"],
                      "es": ["combustible", "gasolineras", "taxis", "transporte"],
                      "fr": ["carburant", "stations-service", "taxis", "transports"]},
    "grocery": {"en": ["grocery shopping", "food shopping", "whole foods", "grocery delivery", "supermarkets",
                       "the weekly shop", "trader joe's"],
                "es": ["comestibles", "mercado", "compras de comida", "despensa"],
                "fr": ["épicerie", "supermarché", "alimentation", "courses alimentaires"]},
    "shopping": {"en": ["retail purchases", "department stores", "shopping trips", "best buy", "target runs",
                        "clothing", "shoes", "gadgets", "mall spending"],
                 "es": ["tiendas", "compras de ropa", "electrónica", "centro comercial"],
                 "fr": ["shopping", "grands magasins", "électronique", "boutiques"]},
    "food_dining": {"en": ["dinners out", "lunch spots", "cafes", "starbucks", "fast food", "bars and restaurants",
                           "food delivery", "doordash orders", "brunch"],
                    "es": ["comida fuera", "cafeterías", "comida rápida", "cenas"],
                    "fr": ["cafés", "repas au restaurant", "restauration rapide", "dîners"]},
    "entertainment": {"en": ["movies", "theater tickets", "events", "video games", "theme parks", "sports tickets",
                             "bowling", "live shows"],
                      "es": ["entretenimiento", "cine", "conciertos", "eventos"],
                      "fr": ["divertissement", "cinéma", "spectacles", "concerts"]},
    "health_fitness": {"en": ["fitness classes", "health expenses", "prescriptions", "medical bills", "dentist",
                              "yoga studio", "drugstore", "cvs and walgreens"],
                       "es": ["farmacia", "salud", "médicos", "clases de yoga"],
                       "fr": ["pharmacie", "santé", "médecin", "fitness"]},
    "home": {"en": ["home depot", "home goods", "household items", "lowe's", "appliances", "garden supplies",
                    "home decor", "repairs around the house"],
             "es": ["hogar", "ferretería", "artículos para el hogar", "decoración"],
             "fr": ["maison", "bricolage", "quincaillerie", "décoration"]},
    "kids_pets": {"en": ["pet food", "veterinarian", "daycare", "toy stores", "baby supplies", "dog grooming",
                         "school supplies", "kids stuff"],
                  "es": ["mascotas", "veterinario", "cosas de niños", "guardería"],
                  "fr": ["animaux", "vétérinaire", "jouets", "garderie"]},
    "personal_care": {"en": ["barber", "spa visits", "nail salon", "cosmetics", "skincare", "beauty salon",
                             "hair salon", "sephora"],
                      "es": ["belleza", "barbería", "cosméticos", "salón de belleza"],
                      "fr": ["beauté", "cosmétiques", "salon de coiffure", "soins"]},
    "travel": {"en": ["plane tickets", "hotels", "airline charges", "car rentals", "vacation bookings",
                      "travel expenses", "trips", "expedia bookings"],
               "es": ["viajes", "boletos de avión", "alquiler de coches", "vacaciones"],
               "fr": ["voyages", "billets d'avion", "location de voiture", "vacances"]},
}
assert set(CATEGORY) == set(CATEGORIES)
CHANNEL = {
    "online": {"en": ["online", "web purchases", "internet orders", "online shopping", "card not present",
                      "ordered online", "on the website", "in-app purchases"],
               "es": ["por internet", "online", "compras web"],
               "fr": ["sur internet", "achats web", "commandes en ligne"]},
    "in_person": {"en": ["in store", "at the register", "point of sale", "swiped in person", "at the terminal",
                         "face to face", "in-person"],
                  "es": ["en persona", "en tienda", "presencial"],
                  "fr": ["en personne", "au magasin", "sur place"]},
}
FRAUD = {
    "fraud": {"en": ["fraud cases", "fraudulent charges", "suspected fraud", "scam transactions", "unauthorized charges",
                     "marked as fraud", "possible fraud"],
              "es": ["fraudulentas", "sospechosas", "no autorizadas"],
              "fr": ["frauduleuses", "suspectes", "non autorisées"]},
    "legit": {"en": ["legitimate", "not fraud", "excluding fraud", "valid transactions", "clean transactions", "no fraud"],
              "es": ["sin fraude", "legítimas", "no fraudulentas"],
              "fr": ["sans fraude", "légitimes", "non frauduleuses"]},
}
AMOUNT = {
    "over": {"en": ["over ${n}", "more than {n} dollars", "above ${n}", "${n} or more", "at least ${n}",
                    "greater than {n} bucks", "over {k}k"],
             "es": ["más de {n} dólares", "por encima de {n} dólares"],
             "fr": ["plus de {n} dollars", "au-dessus de {n} dollars"]},
    "under": {"en": ["under ${n}", "less than {n} dollars", "below ${n}", "${n} or less", "cheaper than {n} bucks"],
              "es": ["menos de {n} dólares", "por debajo de {n} dólares"],
              "fr": ["moins de {n} dollars", "en dessous de {n} dollars"]},
    "between": {"en": ["between ${a} and ${b}", "from ${a} to ${b}", "${a} to ${b}"],
                "es": ["entre {a} y {b} dólares", "de {a} a {b} dólares"],
                "fr": ["entre {a} et {b} dollars", "de {a} à {b} dollars"]},
    "around": {"en": ["around ${n}", "about {n} dollars", "roughly ${n}", "approximately ${n}", "close to {n} bucks"],
               "es": ["alrededor de {n} dólares", "unos {n} dólares"],
               "fr": ["environ {n} euros", "autour de {n} dollars"]},
    "large": {"en": ["large purchases", "big ticket items", "high-value charges", "expensive purchases",
                     "large transactions"],
              "es": ["compras grandes", "cargos altos"],
              "fr": ["gros achats", "montants élevés"]},
    "small": {"en": ["tiny charges", "low-value transactions", "small amounts", "micro purchases", "small purchases"],
              "es": ["compras pequeñas", "montos bajos"],
              "fr": ["petits achats", "petits montants"]},
}
MONTH_NAMES = {"en": ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                      "october", "november"],
               "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                      "octubre", "noviembre"],
               "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
                      "octobre", "novembre"]}
DAY_NAMES = {"en": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
             "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábados", "domingos"],
             "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]}
DATE = {
    "month": {"en": ["in {m}", "during {m}", "from {m}", "back in {m}"], "es": ["en {m}", "durante {m}", "de {m}"],
              "fr": ["en {m}", "pendant {m}", "de {m}"]},
    "weekday": {"en": ["on {d}s", "every {d}", "on {d}", "{d}s"], "es": ["los {d}", "cada {d}"],
                "fr": ["le {d}", "les {d}s", "chaque {d}"]},
    "weekend": {"en": ["on the weekend", "over the weekend", "weekend", "during weekends"],
                "es": ["el fin de semana", "en fin de semana"], "fr": ["en fin de semaine", "pendant le week-end"]},
    "today": {"en": ["from today", "so far today", "made today"], "es": ["hoy", "de hoy"], "fr": ["aujourd'hui"]},
    "yesterday": {"en": ["from yesterday", "made yesterday", "yesterday's"], "es": ["de ayer"], "fr": ["d'hier"]},
    "past_week": {"en": ["in the last 7 days", "over the past week", "this past week", "the last seven days",
                         "last week"], "es": ["últimos 7 días", "esta última semana"],
                  "fr": ["la semaine dernière", "les 7 derniers jours"]},
    "this_month": {"en": ["so far this month", "month to date", "this month"], "es": ["en este mes"], "fr": ["ce mois"]},
    "last_month": {"en": ["previous month", "from last month", "last month"], "es": ["del mes pasado", "mes anterior"],
                   "fr": ["du mois dernier", "le mois précédent"]},
}
SUBJECT = {"en": ["transactions", "charges", "purchases", "payments", "spending", "show me transactions", "find charges",
                  "card activity", "all payments"],
           "es": ["transacciones", "cargos", "compras", "pagos", "gastos"],
           "fr": ["transactions", "achats", "paiements", "dépenses", "opérations"]}
NOTHING = {"en": ["recent activity", "top merchants", "show everything", "sorted by amount", "card ending 4421",
                  "john smith", "latest transactions", "export to csv", "account summary", "monthly statement",
                  "refund status", "merchant list"],
           "es": ["actividad reciente", "todos los movimientos", "resumen de cuenta"],
           "fr": ["activité récente", "tous les mouvements", "relevé mensuel"]}
PREP = {"en": ["in", "from", "cardholders in", "based in", "customers in"], "es": STATE_PREP["es"],
        "fr": STATE_PREP["fr"]}


def blank():
    return {"category": NOT_MENTIONED, "channel": NOT_MENTIONED, "amount": NOT_MENTIONED, "date": NOT_MENTIONED,
            "state_east": NONE, "state_west": NONE, "fraud": NOT_MENTIONED}


def pick(rng, pool, banned):
    allowed = [p for p in pool if p not in banned]
    return rng.choice(allowed or pool)


def fill(rng, tmpl, lang):
    n = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 250, 300, 400, 750, 1500, 2000])
    a = rng.choice([10, 20, 25, 40, 60, 80, 120])
    return tmpl.format(n=n, k=rng.choice([1, 2, 3, 5]), a=a, b=a * rng.choice([2, 3, 5]),
                       m=rng.choice(MONTH_NAMES[lang]), d=rng.choice(DAY_NAMES[lang]))


def segment(rng, field, value, lang, banned):
    """(text, labels) for one filter."""
    if field == "state":
        text = f"{rng.choice(PREP[lang])} {pick(rng, state_phrases(value, lang), banned)}"
        return text, {("state_east" if value in EAST else "state_west"): value}
    if field in ("amount", "date"):
        pool = (AMOUNT if field == "amount" else DATE)[value][lang]
        for _ in range(20):
            text = fill(rng, rng.choice(pool), lang)
            if text not in banned:
                break
        return text, {field: value}
    pools = {"category": CATEGORY, "channel": CHANNEL, "fraud": FRAUD}[field]
    return pick(rng, pools[value][lang], banned), {field: value}


EXTRA = {"state": lambda r: r.choice(list(STATES)), "category": lambda r: r.choice(list(CATEGORY)),
         "channel": lambda r: r.choice(list(CHANNEL)), "amount": lambda r: r.choice(list(AMOUNT)),
         "date": lambda r: r.choice(list(DATE)), "fraud": lambda r: r.choice(list(FRAUD))}


def build_query(rng, required, lang, banned):
    filters = dict(required)
    for field in rng.sample(list(EXTRA), rng.choice([0, 0, 1, 1, 2])):
        filters.setdefault(field, EXTRA[field](rng))
    order = list(filters.items())
    rng.shuffle(order)
    segs = [(rng.choice(SUBJECT[lang]), {})] if rng.random() < 0.5 else []
    return segs + [segment(rng, f, v, lang, banned) for f, v in order]


def cases_from(segs):
    words, spans = [], []
    for text, labels in segs:
        start = len(words)
        words += text.split()
        spans.append((start, len(words), labels))
    out = []
    for k in range(1, len(words) + 1):
        expected, skip = blank(), set()
        for start, end, labels in spans:
            if end <= k:
                expected.update(labels)
            elif start < k:
                skip.update(labels)
        out.append({"state": " ".join(words[:k]), "expected": expected, "skip": sorted(skip), "final": k == len(words)})
    return out


def test_texts():
    """(every test query and prefix, every labelled test phrase that isn't a state's name)."""
    from poc.search.filters import WEST
    names = {n.lower() for n in {**EAST, **WEST}.values()}
    texts, phrases = set(), set()
    for q in parse_queries():
        texts.update(t for t, _ in prefixes(q))
        phrases.update(t for t, labels in q["segments"] if labels and t not in names)
    return texts, phrases


def leaks(cases):
    texts, _ = test_texts()
    return [c["state"] for c in cases if c["state"] in texts]


def generate(seed=11, per_option=18, per_state=6, nothing=60, prefix_share=0.5):
    rng = random.Random(seed)
    banned_texts, banned = test_texts()
    lang = lambda: rng.choices(LANGS, (6, 2, 2))[0]
    queries = [build_query(rng, {"state": code}, lang(), banned) for code in STATES for _ in range(per_state)]
    for field, pool in (("category", CATEGORY), ("channel", CHANNEL), ("amount", AMOUNT), ("date", DATE),
                        ("fraud", FRAUD)):
        for value in pool:
            queries += [build_query(rng, {field: value}, lang(), banned) for _ in range(per_option)]
    for _ in range(nothing):
        lg = lang()
        queries.append([(rng.choice(NOTHING[lg]), {})])
    cases, seen = [], set()
    for segs in queries:
        for case in cases_from(segs):
            if case["state"] in banned_texts or case["state"] in seen:
                continue
            if not case["final"] and rng.random() > prefix_share:
                continue
            seen.add(case["state"])
            cases.append(case)
    return cases


def main():
    cases = generate()
    assert not leaks(cases), leaks(cases)[:5]
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    finals = [c for c in cases if c["final"]]
    print(f"{len(cases)} cases ({len(finals)} full queries, {len(cases) - len(finals)} prefixes) -> {OUT.name}")


if __name__ == "__main__":
    main()
