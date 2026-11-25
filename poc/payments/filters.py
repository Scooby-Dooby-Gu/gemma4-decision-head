"""Payment search filters as decision-head questions, and the step from answers to filters.

The same idea as poc/search/filters.py, on card transactions (Sparkov data, see data.py):
category, online or in person, amount, date, the cardholder's state, and fraud. Each filter is
one multiple-choice question about the typed query, with a "not mentioned" option.

The head picks the kind of amount or date filter ("over", "last month", "on a weekday"); small
parsers read the number, month or weekday off the text, as ages are read in the CRM search.
Relative dates are counted from TODAY, the last day of the generated data.
"""

import calendar
import re
from datetime import date, timedelta

from poc.search.filters import EAST, NONE, NOT_MENTIONED, WEST

TODAY = date(2026, 11, 24)  # a Tuesday; the data runs 2026-01-01 to here
TASK = "the person searching card transactions is filtering by"

CATEGORIES = {
    "gas_transport": "Gas, fuel or getting around (gas stations, rides, transit)",
    "grocery": "Groceries and supermarkets",
    "shopping": "General shopping (clothes, electronics, department stores)",
    "food_dining": "Restaurants, cafes and eating out",
    "entertainment": "Entertainment (movies, concerts, events, games)",
    "health_fitness": "Health and fitness (gym, pharmacy, doctors)",
    "home": "Home (furniture, hardware, home improvement)",
    "kids_pets": "Kids and pets (toys, childcare, pet supplies, vets)",
    "personal_care": "Personal care (hair, beauty, cosmetics)",
    "travel": "Travel (flights, hotels, car rental)",
}
# Sparkov's categories; grocery, shopping and misc come in an online (_net) and in-person (_pos) kind.
SPARKOV = {"grocery": ("grocery_net", "grocery_pos"), "shopping": ("shopping_net", "shopping_pos")}


def _states(states, half):
    criteria = {NONE: "No state from this list, or no place at all"}
    criteria.update(states)
    return {"type": "choice", "instructions": f"Which US state ({half} half) the cardholders live in that {TASK}",
            "criteria": criteria}


QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": f"Which spending category {TASK}",
        "criteria": {NOT_MENTIONED: "No spending category is part of the search", **CATEGORIES},
    },
    "channel": {
        "type": "choice",
        "instructions": f"Whether {TASK} online or in-person transactions",
        "criteria": {
            NOT_MENTIONED: "Online or in person is not part of the search",
            "online": "Online purchases, card not present",
            "in_person": "In person, in a store or at a card terminal",
        },
    },
    "amount": {
        "type": "choice",
        "instructions": f"What kind of amount filter {TASK}",
        "criteria": {
            NOT_MENTIONED: "The amount is not part of the search",
            "over": "More than a given amount",
            "under": "Less than a given amount",
            "between": "Between two given amounts",
            "around": "About one given amount",
            "large": "Large purchases, with no amount given",
            "small": "Small purchases, with no amount given",
        },
    },
    "date": {
        "type": "choice",
        "instructions": f"What kind of date filter {TASK}",
        "criteria": {
            NOT_MENTIONED: "The date is not part of the search",
            "month": "In a named month",
            "weekday": "On a named day of the week",
            "weekend": "On weekends",
            "today": "Today",
            "yesterday": "Yesterday",
            "past_week": "In the last seven days or last week",
            "this_month": "So far this month",
            "last_month": "In the previous calendar month",
        },
    },
    "state_east": _states(EAST, "eastern"),
    "state_west": _states(WEST, "western"),
    "fraud": {
        "type": "choice",
        "instructions": f"Whether {TASK} fraud",
        "criteria": {
            NOT_MENTIONED: "Fraud is not part of the search",
            "fraud": "Fraudulent, suspicious or flagged transactions",
            "legit": "Only legitimate transactions, no fraud",
        },
    },
}

FIELDS = ("category", "channel", "amount_range", "date_value", "state", "fraud")

# --- amounts ------------------------------------------------------------------------------

FIXED_AMOUNT = {"large": [500, None], "small": [None, 20]}
MONEY = re.compile(r"(\$|€)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?\s*(\$|€|dollars?|bucks|usd|dólares|dolares|euros?)?", re.I)


def amounts(text, need=1):
    """Money amounts in the text, in order. If at least `need` numbers are marked as money ($,
    "dollars", "1k"), only those count, so "orders for 2 people over $50" reads 50; otherwise
    every number does ("entre 50 y 100 dólares" marks only the 100)."""
    marked, every = [], []
    for m in MONEY.finditer(text):
        n = float(m.group(2).replace(",", ""))
        if m.group(3):
            n *= 1000
        n = int(n) if n == int(n) else n
        every.append(n)
        if m.group(1) or m.group(3) or m.group(4):
            marked.append(n)
    return marked if len(marked) >= need else every


def amount_range(kind, text):
    """[lo, hi] in dollars for an amount filter kind, or None until the amount is typed."""
    if kind in FIXED_AMOUNT:
        return list(FIXED_AMOUNT[kind])
    nums = amounts(text, need=2 if kind == "between" else 1)
    if not nums:
        return None
    n = nums[0]
    if kind == "over":
        return [n, None]
    if kind == "under":
        return [None, n]
    if kind == "between":
        return sorted(nums[:2]) if len(nums) >= 2 else None
    if kind == "around":
        return [round(n * 0.9, 2), round(n * 1.1, 2)]
    return None


# --- dates ------------------------------------------------------------------------------

MONTHS = {}
for i, names in enumerate([
        "january enero janvier", "february febrero février fevrier", "march marzo mars",
        "april abril avril", "may mayo mai", "june junio juin", "july julio juillet",
        "august agosto août aout", "september septiembre setiembre septembre",
        "october octubre octobre", "november noviembre novembre", "december diciembre décembre decembre"], 1):
    for name in names.split():
        MONTHS[name] = i
WEEKDAYS = {}
for i, names in enumerate([
        "monday mondays lunes lundi lundis", "tuesday tuesdays martes mardi mardis",
        "wednesday wednesdays miércoles miercoles mercredi mercredis",
        "thursday thursdays jueves jeudi jeudis", "friday fridays viernes vendredi vendredis",
        "saturday saturdays sábado sábados sabado sabados samedi samedis",
        "sunday sundays domingo domingos dimanche dimanches"]):
    for name in names.split():
        WEEKDAYS[name] = i


def _span(a, b):
    return f"{a.isoformat()}..{b.isoformat()}"


def date_value(kind, text):
    """The date filter as a comparable string ("2026-10-01..2026-10-31", "dow:4"), or None
    until the month or weekday is typed."""
    words = re.findall(r"[a-záéíóúñèêàçûô]+", text.lower())
    if kind == "month":
        months = [MONTHS[w] for w in words if w in MONTHS]
        if not months:
            return None
        m = months[-1]
        year = TODAY.year if m <= TODAY.month else TODAY.year - 1
        return _span(date(year, m, 1), date(year, m, calendar.monthrange(year, m)[1]))
    if kind == "weekday":
        days = sorted({WEEKDAYS[w] for w in words if w in WEEKDAYS})
        return "dow:" + ",".join(map(str, days)) if days else None
    if kind == "weekend":
        return "dow:5,6"
    if kind == "today":
        return _span(TODAY, TODAY)
    if kind == "yesterday":
        y = TODAY - timedelta(days=1)
        return _span(y, y)
    if kind == "past_week":
        return _span(TODAY - timedelta(days=6), TODAY)
    if kind == "this_month":
        return _span(TODAY.replace(day=1), TODAY)
    if kind == "last_month":
        end = TODAY.replace(day=1) - timedelta(days=1)
        return _span(end.replace(day=1), end)
    return None


# --- answers -> filters -------------------------------------------------------------------

def gate(answer, threshold):
    if answer["abstain"] or answer["confidence"] < threshold:
        return None
    return answer["choice"]


def to_filters(text, answers, threshold=0.5):
    """{"category", "channel", "amount", "amount_range", "date", "date_value", "state", "fraud"}."""
    picks = {k: gate(answers[k], threshold) for k in ("category", "channel", "amount", "date", "fraud")}
    picks = {k: None if v == NOT_MENTIONED else v for k, v in picks.items()}
    states = [(answers[k]["confidence"], gate(answers[k], threshold)) for k in ("state_east", "state_west")]
    states = [(c, s) for c, s in states if s and s != NONE]
    rng = amount_range(picks["amount"], text) if picks["amount"] else None
    when = date_value(picks["date"], text) if picks["date"] else None
    return {
        "category": picks["category"],
        "channel": picks["channel"],
        "amount": picks["amount"] if rng else None,  # the kind is clear but no amount yet: wait
        "amount_range": rng,
        "date": picks["date"] if when else None,
        "date_value": when,
        "state": max(states)[1] if states else None,
        "fraud": picks["fraud"],
    }


def where(filters):
    """A parameterised WHERE clause over the transactions table: (sql, params)."""
    parts, params = [], []
    cat, channel = filters.get("category"), filters.get("channel")
    if cat:
        cats = SPARKOV.get(cat, (cat,))
        parts.append(f"category IN ({','.join('?' * len(cats))})")
        params += cats
    if channel == "online":
        parts.append("category LIKE '%\\_net' ESCAPE '\\'")
    elif channel == "in_person":
        parts.append("category NOT LIKE '%\\_net' ESCAPE '\\'")
    if filters.get("amount_range"):
        lo, hi = filters["amount_range"]
        if lo is not None:
            parts.append("amount >= ?")
            params.append(lo)
        if hi is not None:
            parts.append("amount <= ?")
            params.append(hi)
    v = filters.get("date_value")
    if v and v.startswith("dow:"):
        days = [int(d) for d in v[4:].split(",")]
        # SQLite: strftime('%w') is 0 = Sunday; Python weekday() is 0 = Monday.
        parts.append(f"CAST(strftime('%w', date) AS INTEGER) IN ({','.join('?' * len(days))})")
        params += [(d + 1) % 7 for d in days]
    elif v:
        lo, hi = v.split("..")
        parts.append("date BETWEEN ? AND ?")
        params += [lo, hi]
    if filters.get("state"):
        parts.append("state = ?")
        params.append(filters["state"])
    if filters.get("fraud"):
        parts.append("is_fraud = ?")
        params.append(1 if filters["fraud"] == "fraud" else 0)
    return " AND ".join(parts) or "1", params
