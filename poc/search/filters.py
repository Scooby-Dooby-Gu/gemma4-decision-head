"""CRM search filters as decision-head questions, and the step from answers to applied filters.

Each filter is one question about the search query. Every question has a "nothing asked for"
option so a filter stays off when the query doesn't mention it. A filter is applied only when
the head picks something other than that option, doesn't abstain, and is confident enough.

Two things the head can't do on its own, handled here:
  - 50 states plus DC is more than the ~34 answer handles a question can have, so states are
    split into an east and a west question, each with a "none" option.
  - The head picks the kind of age filter ("older than", "in their forties"), but it can't read
    an arbitrary number off a handle. A small parser pulls the number out of the query text.
"""

import re

NONE = "none"
NOT_MENTIONED = "not_mentioned"

# East of the Mississippi (plus DC), and west of it, split so each list fits the handles.
EAST = {
    "AL": "Alabama", "CT": "Connecticut", "DC": "Washington DC", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "IL": "Illinois", "IN": "Indiana", "KY": "Kentucky",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MS": "Mississippi", "NH": "New Hampshire", "NJ": "New Jersey", "NY": "New York",
    "NC": "North Carolina", "OH": "Ohio", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "TN": "Tennessee", "VT": "Vermont", "VA": "Virginia",
    "WV": "West Virginia", "WI": "Wisconsin",
}
WEST = {
    "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "HI": "Hawaii", "ID": "Idaho", "IA": "Iowa", "KS": "Kansas", "LA": "Louisiana",
    "MN": "Minnesota", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NM": "New Mexico", "ND": "North Dakota", "OK": "Oklahoma", "OR": "Oregon",
    "SD": "South Dakota", "TX": "Texas", "UT": "Utah", "WA": "Washington state", "WY": "Wyoming",
}

TASK = "the person searching a client database is filtering by"


def _states(states, half):
    criteria = {NONE: "No state from this list, or no place at all"}
    criteria.update(states)
    return {"type": "choice", "instructions": f"Which US state ({half} half) {TASK}", "criteria": criteria}


QUESTIONS = {
    "state_east": _states(EAST, "eastern"),
    "state_west": _states(WEST, "western"),
    "kids": {
        "type": "choice",
        "instructions": f"Whether {TASK} clients having children",
        "criteria": {
            NOT_MENTIONED: "Children are not part of the search",
            "has_kids": "Wants clients who have children",
            "no_kids": "Wants clients who have no children",
        },
    },
    "age": {
        "type": "choice",
        "instructions": f"What kind of age filter {TASK} for the clients themselves",
        "criteria": {
            NOT_MENTIONED: "The clients' age is not part of the search",
            "older_than": "Older than a given age",
            "younger_than": "Younger than a given age",
            "between": "Between two given ages",
            "around": "About one given age",
            "decade": "In a given decade of life",
            "young": "Young adults, with no number given",
            "middle_aged": "Middle-aged, with no number given",
            "seniors": "Older people or retirees, with no number given",
        },
    },
    "marital": {
        "type": "choice",
        "instructions": f"Which marital status {TASK}",
        "criteria": {
            NOT_MENTIONED: "Marital status is not part of the search",
            "married": "Married",
            "single": "Single, never married",
            "divorced": "Divorced",
            "widowed": "Widowed",
        },
    },
}

# --- age numbers ------------------------------------------------------------------------

TENS = {
    # English
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60, "seventies": 70,
    "twentysomething": 20, "twentysomethings": 20, "thirtysomething": 30, "thirtysomethings": 30,
    "fortysomething": 40, "fortysomethings": 40, "fiftysomething": 50, "fiftysomethings": 50,
    # Spanish
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "veintes": 20, "treintas": 30, "cuarentas": 40, "cincuentas": 50, "sesentas": 60,
    # French
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60, "soixante-dix": 70,
    "vingtaine": 20, "trentaine": 30, "quarantaine": 40, "cinquantaine": 50, "soixantaine": 60,
    "trentenaires": 30, "quadragénaires": 40, "quinquagénaires": 50, "sexagénaires": 60,
}
INCLUSIVE_UP = re.compile(r"\+|at least|or older|and up|and over|o más|ou plus|au moins|al menos", re.I)
INCLUSIVE_DOWN = re.compile(r"at most|or younger|and under|o menos|ou moins|como máximo", re.I)
FIXED = {"young": (18, 34), "middle_aged": (40, 59), "seniors": (65, None)}


def numbers(text):
    """Ages written in the text, in order: digits ("40", "40s", "50+") or words ("forty")."""
    out = []
    for tok in re.findall(r"\d+|[a-záéíóúñèêàç-]+", text.lower()):
        if tok.isdigit():
            if 1 <= int(tok) <= 120:
                out.append(int(tok))
        elif tok in TENS:
            out.append(TENS[tok])
    return out


def age_range(kind, text):
    """[lo, hi] (either may be None) for an age filter kind, or None if the number isn't typed yet."""
    if kind in FIXED:
        return list(FIXED[kind])
    nums = numbers(text)
    if not nums:
        return None
    n = nums[0]
    if kind == "older_than":
        return [n if INCLUSIVE_UP.search(text) else n + 1, None]
    if kind == "younger_than":
        return [None, n if INCLUSIVE_DOWN.search(text) else n - 1]
    if kind == "between":
        return sorted(nums[:2]) if len(nums) >= 2 else None
    if kind == "decade":
        return [n // 10 * 10, n // 10 * 10 + 9]
    if kind == "around":
        return [n - 2, n + 2]
    return None


# --- answers -> applied filters ---------------------------------------------------------

def gate(answer, threshold):
    """The head's choice if it is confident and doesn't abstain, else None."""
    if answer["abstain"] or answer["confidence"] < threshold:
        return None
    return answer["choice"]


def to_filters(text, answers, threshold=0.5):
    """{"state", "kids", "age", "age_range", "marital"}: the filters to apply, None where off."""
    states = []
    for key in ("state_east", "state_west"):
        pick = gate(answers[key], threshold)
        if pick and pick != NONE:
            states.append((answers[key]["confidence"], pick))
    kids = gate(answers["kids"], threshold)
    age = gate(answers["age"], threshold)
    marital = gate(answers["marital"], threshold)
    age = None if age == NOT_MENTIONED else age
    rng = age_range(age, text) if age else None
    if age and rng is None:
        age = None  # the kind is clear but the number isn't typed yet: wait for it
    return {
        "state": max(states)[1] if states else None,
        "kids": None if kids == NOT_MENTIONED else kids,
        "age": age,
        "age_range": rng,
        "marital": None if marital == NOT_MENTIONED else marital,
    }


def where_clause(filters):
    """The SQL the filters turn into (for the demo; phase 0 only shows it)."""
    parts = []
    if filters["state"]:
        parts.append(f"state = '{filters['state']}'")
    if filters["kids"] == "has_kids":
        parts.append("kids > 0")
    elif filters["kids"] == "no_kids":
        parts.append("kids = 0")
    if filters["age_range"]:
        lo, hi = filters["age_range"]
        if lo is not None:
            parts.append(f"age >= {lo}")
        if hi is not None:
            parts.append(f"age <= {hi}")
    if filters["marital"]:
        parts.append(f"marital = '{filters['marital']}'")
    return " AND ".join(parts) or "TRUE"
