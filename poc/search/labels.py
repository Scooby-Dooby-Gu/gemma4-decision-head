"""Training cases for the search filters, generated from phrase pools in English, Spanish and French.

Hand-writing 20 queries for each of the ~60 options isn't practical, so each option gets a pool
of short phrases, and queries are built by combining phrases with filler words ("clients",
"living in", "show me"). Every query is also cut into its word-by-word prefixes, labelled with
only the filters finished so far, so the head learns to wait: "solteras" alone means single,
and says nothing about kids.

Kept apart from the test set (queries.txt):
  - no training query or prefix is the same text as any test query or prefix (`leaks()` checks);
  - no phrase the test set labels as a segment ("families", "the big apple", "nola", "dads") is
    picked from the pools, except a state's own English name ("texas", "virginia"), which is the
    option's own wording;
  - no deliberate look-alikes of the test traps ("with kids over 10", "georgia phone number").
Cities come from crm.py, minus names that belong to several states (Portland, Charleston).

    python -m poc.search.labels          # writes train_cases.json and prints counts
"""

import json
import random
from collections import Counter
from pathlib import Path

from .crm import STATES
from .filters import EAST, NONE, NOT_MENTIONED, WEST
from .measure import parse_queries, prefixes

HERE = Path(__file__).resolve().parent
OUT = HERE / "train_cases.json"
LANGS = ("en", "es", "fr")

# --- phrase pools -------------------------------------------------------------------------

AMBIGUOUS_CITIES = {"Portland", "Charleston", "Arlington", "Jackson", "Columbia", "Springfield",
                    "Washington", "Athens", "Kansas City", "Columbus"}
STATE_EXTRA = {  # nicknames, abbreviations people type, and Spanish / French names
    "NY": {"en": ["manhattan", "upstate new york", "ny state"], "es": ["el estado de nueva york"], "fr": ["l'état de new york"]},
    "CA": {"en": ["southern california", "northern california", "cali", "silicon valley"], "fr": ["la californie"]},
    "TX": {"en": ["tx", "dfw"], "es": ["tejas"]},
    "FL": {"en": ["south florida", "fla"], "es": ["la florida"]},
    "PA": {"en": ["pittsburgh area"], "es": ["pensilvania"]},
    "IL": {"en": ["the windy city"]},
    "LA": {"en": ["the big easy", "baton rouge area"], "es": ["luisiana"], "fr": ["la louisiane", "louisiane", "la nouvelle-orléans"]},
    "MA": {"en": ["beantown", "cape cod"], "fr": ["le massachusetts"]},
    "NJ": {"en": ["jersey", "the garden state"], "fr": ["le new jersey"]},
    "NC": {"en": ["nc", "the research triangle"], "es": ["carolina del norte"], "fr": ["caroline du nord"]},
    "SC": {"en": ["sc"], "es": ["carolina del sur"], "fr": ["caroline du sud"]},
    "VA": {"en": ["northern virginia", "nova"], "fr": ["la virginie", "virginie"]},
    "WV": {"en": ["wv"], "es": ["virginia occidental"], "fr": ["virginie-occidentale"]},
    "NM": {"en": ["nm"], "fr": ["le nouveau-mexique", "nouveau-mexique"]},
    "WA": {"en": ["wa state", "puget sound"], "es": ["el estado de washington"], "fr": ["l'état de washington"]},
    "DC": {"en": ["d.c.", "the district"], "es": ["washington d.c."], "fr": ["washington d.c."]},
    "GA": {"en": ["the peach state"], "fr": ["la géorgie"]},
    "NV": {"en": ["sin city", "reno area"]},
    "HI": {"en": ["oahu", "maui"], "es": ["hawái"], "fr": ["hawaï"]},
    "AZ": {"en": ["az"], "fr": ["l'arizona"]},
    "CO": {"en": ["the front range"], "fr": ["le colorado"]},
    "MI": {"en": ["the mitten state"], "es": ["míchigan"]},
    "MO": {"en": ["the show me state"], "es": ["misuri"]},
    "MN": {"en": ["the twin cities"], "fr": ["le minnesota"]},
    "TN": {"en": ["music city"], "fr": ["le tennessee"]},
    "OR": {"en": ["oregon state"], "es": ["oregón"]},
}
STATE_PREP = {"en": ["in", "from", "living in", "based in", "who live in", "located in"],
              "es": ["en", "de", "que viven en", "ubicados en"],
              "fr": ["en", "à", "au", "qui habitent en", "basés en"]}

KIDS = {
    "has_kids": {"en": ["who have kids", "with a child", "with young kids", "moms and dads", "mothers", "fathers",
                        "with toddlers", "who are parents", "with sons or daughters", "with grown children",
                        "with little ones", "dads"],
                 "es": ["que tienen hijos", "con niños pequeños", "mamás", "papás", "con un hijo", "que son padres"],
                 "fr": ["qui ont des enfants", "mamans", "papas", "avec des enfants", "avec un enfant", "pères de famille"]},
    "no_kids": {"en": ["with no children", "who don't have kids", "zero children", "not parents", "kid free",
                       "childfree", "without any kids", "no children at all"],
                "es": ["que no tienen hijos", "sin descendencia", "sin ningún hijo"],
                "fr": ["sans enfant", "qui n'ont pas d'enfants", "sans aucun enfant"]},
}
MARITAL = {
    "married": {"en": ["married couples", "husbands and wives", "who are married", "spouses", "married people"],
                "es": ["casadas", "matrimonios", "que están casados"], "fr": ["mariées", "gens mariés", "qui sont mariés"]},
    "single": {"en": ["unmarried", "never married", "singles", "single people", "who aren't married"],
               "es": ["solteras", "gente soltera", "que no están casados"], "fr": ["célibataire", "pas mariés", "personnes seules"]},
    "divorced": {"en": ["divorcees", "divorced women", "who got divorced", "divorced men", "recently divorced"],
                 "es": ["divorciadas", "que se divorciaron"], "fr": ["divorcée", "divorcées", "qui ont divorcé"]},
    "widowed": {"en": ["widowers", "widow", "who lost a spouse", "widowed women"],
                "es": ["viudos", "viudo"], "fr": ["veuves", "veuve", "qui ont perdu leur conjoint"]},
}
# Age phrases: {n} is a number, {a} {b} two numbers, {w} a decade word.
AGE = {
    "older_than": {"en": ["over {n}", "older than {n}", "above {n}", "more than {n} years old", "{n}+", "{n} and older",
                          "{n} or older", "at least {n}", "aged {n} and up"],
                   "es": ["con más de {n} años", "de más de {n}", "mayores de {n} años"],
                   "fr": ["de plus de {n} ans", "plus de {n}", "{n} ans et plus", "âgés de plus de {n} ans"]},
    "younger_than": {"en": ["under {n}", "younger than {n}", "below {n}", "less than {n} years old", "{n} or younger",
                            "{n} and under", "not yet {n}"],
                     "es": ["menores de {n} años", "con menos de {n} años", "de menos de {n}"],
                     "fr": ["de moins de {n} ans", "moins de {n}", "âgés de moins de {n} ans"]},
    "between": {"en": ["between {a} and {b}", "aged {a} to {b}", "from {a} to {b} years old", "{a}-{b} years old",
                       "{a} to {b}"],
                "es": ["entre {a} y {b} años", "de {a} a {b}"], "fr": ["entre {a} et {b} ans", "de {a} à {b} ans"]},
    "around": {"en": ["around {n}", "about {n} years old", "roughly {n}", "{n}-year-olds", "approximately {n}"],
               "es": ["de unos {n} años", "de alrededor de {n} años"], "fr": ["d'environ {n} ans", "autour de {n} ans"]},
    "decade": {"en": ["in their {n}s", "people in their {w}", "{w}something", "in their late {n}s", "in their early {n}s"],
               "es": ["de {w} y tantos", "que rondan los {w}", "en sus {n}s"],
               "fr": ["dans la {fw}", "{fn}", "dans leur {fw}"]},
    "young": {"en": ["young adults", "younger clients", "recent grads", "young people"],
              "es": ["gente joven", "adultos jóvenes"], "fr": ["jeunes adultes", "jeunes gens"]},
    "middle_aged": {"en": ["middle-aged", "in midlife", "middle age"],
                    "es": ["de mediana edad", "de edad media"], "fr": ["d'âge mûr", "d'âge moyen"]},
    "seniors": {"en": ["elderly", "older adults", "retired", "pensioners", "senior citizens", "the elderly"],
                "es": ["personas mayores", "de la tercera edad", "pensionados"], "fr": ["personnes âgées", "aînés", "seniors à la retraite"]},
}
DECADE_WORDS = {"en": {30: "thirties", 40: "forties", 50: "fifties", 60: "sixties"},
                "es": {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta"}}
DECADE_FR = {30: ("trentaine", "trentenaires"), 40: ("quarantaine", "quadragénaires"),
             50: ("cinquantaine", "quinquagénaires"), 60: ("soixantaine", "sexagénaires")}
EN_SOMETHING = {30: "thirty", 40: "forty", 50: "fifty"}

SUBJECT = {"en": ["clients", "people", "contacts", "customers", "leads", "show me clients", "find people", "anyone"],
           "es": ["clientes", "personas", "contactos", "gente", "busca clientes"],
           "fr": ["clients", "personnes", "contacts", "trouve des clients"]}
NOTHING = {"en": ["maria lopez", "recently contacted", "last month", "biggest accounts", "follow up list", "teachers",
                  "accountants", "who called yesterday", "from the referral list", "vip list", "inactive clients",
                  "emails bounced", "david chen", "show all", "sorted by name"],
           "es": ["contactados recientemente", "lista de seguimiento", "cuentas grandes", "maestros", "juan garcía"],
           "fr": ["contactés récemment", "liste de suivi", "grands comptes", "enseignants", "marie dubois"]}


def state_phrases(code, lang):
    name = {**EAST, **WEST}[code].lower()
    out = [name] if lang == "en" or code not in ("NY", "NJ", "NM", "NC", "SC", "PA", "LA", "WV") else []
    if lang == "en":
        out += [c.lower() for c in STATES[code][1] if c not in AMBIGUOUS_CITIES]
    out += STATE_EXTRA.get(code, {}).get(lang, [])
    if lang != "en" and not out:
        out = [name]
    return out


def age_phrase(rng, kind, lang):
    tmpl = rng.choice(AGE[kind][lang])
    decade = rng.choice([30, 40, 50, 60])
    n = rng.randint(21, 75)
    a = rng.randint(22, 55)
    b = a + rng.randint(5, 20)
    if kind == "decade":
        if lang == "en" and "something" in tmpl:
            decade = rng.choice(list(EN_SOMETHING))
            return tmpl.format(w=EN_SOMETHING[decade])
        fw, fn = DECADE_FR[decade]
        return tmpl.format(n=decade, w=DECADE_WORDS.get(lang, {}).get(decade, ""), fw=fw, fn=fn)
    return tmpl.format(n=n, a=a, b=b)


# --- building queries -------------------------------------------------------------------

def blank():
    return {"state_east": NONE, "state_west": NONE, "kids": NOT_MENTIONED, "age": NOT_MENTIONED,
            "marital": NOT_MENTIONED}


def pick(rng, pool, banned):
    """A phrase from the pool that the test set doesn't label (or any, if all are)."""
    allowed = [p for p in pool if p not in banned]
    return rng.choice(allowed or pool)


def segment(rng, field, value, lang, banned):
    """(text, labels) for one filter."""
    if field == "state":
        text = f"{rng.choice(STATE_PREP[lang])} {pick(rng, state_phrases(value, lang), banned)}"
        return text, {("state_east" if value in EAST else "state_west"): value}
    if field == "kids":
        return pick(rng, KIDS[value][lang], banned), {"kids": value}
    if field == "marital":
        return pick(rng, MARITAL[value][lang], banned), {"marital": value}
    for _ in range(20):
        text = age_phrase(rng, value, lang)
        if text not in banned:
            break
    return text, {"age": value}


def build_query(rng, required, lang, banned):
    """A query with the `required` (field, value) filters, maybe one or two more, in a random order."""
    extra = {"state": lambda: rng.choice(list(STATES)), "kids": lambda: rng.choice(list(KIDS)),
             "marital": lambda: rng.choice(list(MARITAL)), "age": lambda: rng.choice(list(AGE))}
    filters = dict(required)
    for field in rng.sample(list(extra), rng.choice([0, 0, 1, 1, 2])):
        filters.setdefault(field, extra[field]())
    order = list(filters.items())
    rng.shuffle(order)
    segs = []
    if rng.random() < 0.6:
        segs.append((rng.choice(SUBJECT[lang]), {}))
    for field, value in order:
        segs.append(segment(rng, field, value, lang, banned))
    return segs


def cases_from(segs):
    """The full query and every word-by-word prefix, each labelled with the filters finished so far.

    A phrase still being typed may already mean something ("married" in "married couples",
    "more than 40" in "more than 40 years old"), so its question gets no label from that prefix
    ("skip"): calling it "not mentioned" would teach the head to ignore a complete filter. The
    prefix still labels every other question."""
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
        out.append({"state": " ".join(words[:k]), "expected": expected, "skip": sorted(skip),
                     "final": k == len(words)})
    return out


def generate(seed=11, per_option=14, per_state=10, nothing=60, prefix_share=0.5):
    rng = random.Random(seed)
    banned_texts, banned = test_texts()
    queries = []
    for code in STATES:
        for _ in range(per_state):
            queries.append(build_query(rng, {"state": code}, rng.choices(LANGS, (6, 2, 2))[0], banned))
    for field, pool in (("kids", KIDS), ("marital", MARITAL), ("age", AGE)):
        for value in pool:
            for _ in range(per_option):
                queries.append(build_query(rng, {field: value}, rng.choices(LANGS, (6, 2, 2))[0], banned))
    for _ in range(nothing):
        lang = rng.choices(LANGS, (6, 2, 2))[0]
        queries.append([(rng.choice(NOTHING[lang]), {})])
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


def test_texts():
    """(every test query and prefix text, every labelled test segment that isn't a state's name)."""
    names = {n.lower() for n in {**EAST, **WEST}.values()}
    texts, phrases = set(), set()
    for q in parse_queries(HERE / "queries.txt"):
        texts.update(t for t, _ in prefixes(q))
        phrases.update(t for t, labels in q["segments"] if labels and t not in names)
    return texts, phrases


def leaks(cases):
    """Training texts that are the same as a test query or prefix (should be none)."""
    texts, _ = test_texts()
    return [c["state"] for c in cases if c["state"] in texts]


def main():
    cases = generate()
    assert not leaks(cases), leaks(cases)[:5]
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    finals = [c for c in cases if c["final"]]
    print(f"{len(cases)} cases ({len(finals)} full queries, {len(cases) - len(finals)} prefixes) -> {OUT.name}")
    print(f"  {sum(bool(c['skip']) for c in cases)} prefixes leave the phrase being typed unlabelled")
    for key in blank():
        counts = Counter(c["expected"][key] for c in cases if key not in c["skip"])
        print(f"  {key}: {len(counts)} options, fewest {min(counts.values())} ({min(counts, key=counts.get)}), "
              f"'{blank()[key]}' {counts[blank()[key]]}")
    for c in random.Random(1).sample(cases, 12):
        print("   ", c["state"], {k: v for k, v in c["expected"].items() if v not in (NONE, NOT_MENTIONED)})


if __name__ == "__main__":
    main()
