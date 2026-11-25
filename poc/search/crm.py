"""A made-up CRM to search: 5,000 client profiles in SQLite, the same every time (seeded).

Every person is invented: names are drawn from short lists of common first and last names and
paired at random. The table has the fields the filters use (state, age, kids, marital) plus a
city, occupation and last-contact date so the results look like a CRM.
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .filters import EAST, WEST

DB = Path(__file__).resolve().parent / "crm.db"

FIRST = """James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara
Richard Susan Joseph Jessica Thomas Sarah Charles Karen Daniel Lisa Matthew Nancy Anthony Betty
Mark Sandra Steven Ashley Andrew Emily Joshua Michelle Kevin Amanda Brian Melissa Jose Maria
Carlos Sofia Luis Camila Wei Mei Hiroshi Yuki Priya Arjun Fatima Omar Chloe Lucas Amara Kwame
Noah Olivia Liam Emma Ethan Ava Mateo Isabella Elijah Mia""".split()
LAST = """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez
Gonzalez Wilson Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez
Clark Ramirez Lewis Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill Flores Green
Adams Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts Chen Patel Kim Okafor Cohen
Murphy Rossi Dubois Silva""".split()
OCCUPATIONS = ["Teacher", "Nurse", "Software engineer", "Accountant", "Attorney", "Physician",
               "Contractor", "Realtor", "Small business owner", "Pharmacist", "Sales manager",
               "Architect", "Dentist", "Professor", "Marketing director", "Financial analyst",
               "Electrician", "Chef", "Consultant"]

# Rough population weights (millions) and a few cities, so big states show up more often.
STATES = {
    "CA": (39, ["Los Angeles", "San Francisco", "San Diego", "Oakland", "Sacramento"]),
    "TX": (30, ["Houston", "Austin", "Dallas", "San Antonio"]),
    "FL": (22, ["Miami", "Orlando", "Tampa", "Jacksonville"]),
    "NY": (20, ["New York", "Brooklyn", "Buffalo", "Rochester"]),
    "PA": (13, ["Philadelphia", "Pittsburgh", "Allentown"]),
    "IL": (12.5, ["Chicago", "Naperville", "Springfield"]),
    "OH": (11.8, ["Columbus", "Cleveland", "Cincinnati"]),
    "GA": (11, ["Atlanta", "Savannah", "Athens"]),
    "NC": (10.7, ["Charlotte", "Raleigh", "Durham"]),
    "MI": (10, ["Detroit", "Grand Rapids", "Ann Arbor"]),
    "NJ": (9.3, ["Newark", "Jersey City", "Princeton"]),
    "VA": (8.7, ["Richmond", "Arlington", "Norfolk"]),
    "WA": (7.8, ["Seattle", "Spokane", "Tacoma"]),
    "AZ": (7.4, ["Phoenix", "Tucson", "Scottsdale"]),
    "TN": (7, ["Nashville", "Memphis", "Knoxville"]),
    "MA": (7, ["Boston", "Cambridge", "Worcester"]),
    "IN": (6.8, ["Indianapolis", "Fort Wayne"]),
    "MO": (6.2, ["St. Louis", "Kansas City"]),
    "MD": (6.2, ["Baltimore", "Bethesda"]),
    "WI": (5.9, ["Milwaukee", "Madison"]),
    "CO": (5.9, ["Denver", "Boulder", "Colorado Springs"]),
    "MN": (5.7, ["Minneapolis", "St. Paul"]),
    "SC": (5.3, ["Charleston", "Columbia"]),
    "AL": (5.1, ["Birmingham", "Montgomery"]),
    "LA": (4.6, ["New Orleans", "Baton Rouge"]),
    "KY": (4.5, ["Louisville", "Lexington"]),
    "OR": (4.2, ["Portland", "Eugene"]),
    "OK": (4, ["Oklahoma City", "Tulsa"]),
    "CT": (3.6, ["Hartford", "New Haven", "Stamford"]),
    "UT": (3.4, ["Salt Lake City", "Provo"]),
    "IA": (3.2, ["Des Moines", "Cedar Rapids"]),
    "NV": (3.2, ["Las Vegas", "Reno"]),
    "AR": (3, ["Little Rock", "Fayetteville"]),
    "MS": (2.9, ["Jackson", "Gulfport"]),
    "KS": (2.9, ["Wichita", "Overland Park"]),
    "NM": (2.1, ["Albuquerque", "Santa Fe"]),
    "NE": (2, ["Omaha", "Lincoln"]),
    "ID": (1.9, ["Boise"]),
    "WV": (1.8, ["Charleston", "Morgantown"]),
    "HI": (1.4, ["Honolulu"]),
    "NH": (1.4, ["Manchester", "Portsmouth"]),
    "ME": (1.4, ["Portland", "Bangor"]),
    "RI": (1.1, ["Providence"]),
    "MT": (1.1, ["Billings", "Missoula"]),
    "DE": (1, ["Wilmington", "Dover"]),
    "SD": (0.9, ["Sioux Falls"]),
    "ND": (0.8, ["Fargo"]),
    "AK": (0.7, ["Anchorage"]),
    "DC": (0.7, ["Washington"]),
    "VT": (0.6, ["Burlington"]),
    "WY": (0.6, ["Cheyenne"]),
}
assert set(STATES) == set(EAST) | set(WEST)


def profile(rng, i):
    state = rng.choices(list(STATES), weights=[w for w, _ in STATES.values()])[0]
    age = max(22, min(88, int(rng.gauss(48, 15))))
    if age >= 70:
        marital = rng.choices(["married", "widowed", "divorced", "single"], [50, 30, 12, 8])[0]
    elif age >= 35:
        marital = rng.choices(["married", "divorced", "single", "widowed"], [58, 18, 21, 3])[0]
    else:
        marital = rng.choices(["single", "married", "divorced"], [58, 37, 5])[0]
    has_kids_p = {"married": 0.75, "divorced": 0.65, "widowed": 0.8, "single": 0.2}[marital]
    if age < 28:
        has_kids_p *= 0.5
    kids = rng.choices([1, 2, 3, 4], [35, 40, 18, 7])[0] if rng.random() < has_kids_p else 0
    occupation = "Retired" if age >= 67 and rng.random() < 0.8 else rng.choice(OCCUPATIONS)
    contacted = date(2026, 11, 20) - timedelta(days=int(rng.expovariate(1 / 60)))
    return (i, f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng.choice(STATES[state][1]), state, age,
            kids, marital, occupation, contacted.isoformat())


def build(path=DB, n=5000, seed=7):
    path = Path(path)
    path.unlink(missing_ok=True)
    rng = random.Random(seed)
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE clients (id INTEGER PRIMARY KEY, name TEXT, city TEXT, state TEXT,
                      age INTEGER, kids INTEGER, marital TEXT, occupation TEXT, last_contact TEXT)""")
        db.executemany("INSERT INTO clients VALUES (?,?,?,?,?,?,?,?,?)",
                       [profile(rng, i + 1) for i in range(n)])
        for col in ("state", "age", "kids", "marital"):
            db.execute(f"CREATE INDEX idx_{col} ON clients({col})")
    return path


def connect(path=DB):
    if not Path(path).exists():
        build(path)
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def query(filters):
    """(where SQL with ? placeholders, params) for applied filters (see filters.to_filters)."""
    parts, params = [], []
    if filters.get("state"):
        parts.append("state = ?")
        params.append(filters["state"])
    if filters.get("kids") == "has_kids":
        parts.append("kids > 0")
    elif filters.get("kids") == "no_kids":
        parts.append("kids = 0")
    lo, hi = filters.get("age_range") or (None, None)
    if lo is not None:
        parts.append("age >= ?")
        params.append(lo)
    if hi is not None:
        parts.append("age <= ?")
        params.append(hi)
    if filters.get("marital"):
        parts.append("marital = ?")
        params.append(filters["marital"])
    return " AND ".join(parts) or "1", params


def search(db, filters, limit=50):
    """{"total", "count", "rows", "where"} for the filters, most recently contacted first."""
    where, params = query(filters)
    count = db.execute(f"SELECT COUNT(*) FROM clients WHERE {where}", params).fetchone()[0]
    rows = db.execute(f"SELECT * FROM clients WHERE {where} ORDER BY last_contact DESC LIMIT ?",
                      params + [limit]).fetchall()
    total = db.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
    shown = where
    for p in params:
        shown = shown.replace("?", repr(p), 1)
    return {"total": total, "count": count, "rows": [dict(r) for r in rows], "where": shown}
