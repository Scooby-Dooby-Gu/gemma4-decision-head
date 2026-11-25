"""Load Sparkov's generated card transactions into SQLite (poc/payments/txn.db, not committed).

Sparkov (https://github.com/namebrandon/Sparkov_Data_Generation, MIT) is run separately, in its
own folder and virtual environment, at commit b5eb45c:

    git clone https://github.com/namebrandon/Sparkov_Data_Generation .vendor/sparkov
    cd .vendor/sparkov && python3.13 -m venv .venv && .venv/bin/pip install Faker numpy
    .venv/bin/python datagen.py -n 300 -seed 7 -o out 01-01-2026 11-24-2026

then, from the repo root:

    python -m poc.payments.data .vendor/sparkov/out

300 cardholders, 227,468 transactions, 2026-01-01 to 2026-11-24. The fake SSNs, card and
account numbers and street addresses Sparkov writes are left out.
"""

import csv
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "txn.db"
KEEP = ("trans_num", "trans_date", "trans_time", "category", "amt", "is_fraud", "merchant",
        "first", "last", "gender", "city", "state", "dob")


def build(src, path=DB):
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE txn (id TEXT PRIMARY KEY, date TEXT, time TEXT, category TEXT, amount REAL,
                  is_fraud INTEGER, merchant TEXT, first TEXT, last TEXT, gender TEXT, city TEXT,
                  state TEXT, dob TEXT)""")
    n = 0
    for f in sorted(Path(src).glob("*.csv")):
        with open(f, newline="") as fh:
            rows = csv.DictReader(fh, delimiter="|")
            if not rows.fieldnames or "category" not in rows.fieldnames:
                continue  # Sparkov leaves an empty file when a chunk gets no customers
            batch = []
            for r in rows:
                v = [r[k] for k in KEEP]
                v[6] = v[6].removeprefix("fraud_")  # Sparkov prefixes every merchant name
                batch.append(v)
            db.executemany(f"INSERT INTO txn VALUES ({','.join('?' * len(KEEP))})", batch)
            n += len(batch)
    for col in ("date", "category", "state", "amount"):
        db.execute(f"CREATE INDEX idx_{col} ON txn({col})")
    db.commit()
    return db, n


def connect():
    if not DB.exists():
        raise SystemExit(f"{DB} is missing: generate Sparkov data and run python -m poc.payments.data <folder>")
    db = sqlite3.connect(DB, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def search(db, filters, limit=50):
    from .filters import where

    sql, params = where(filters)
    total = db.execute(f"SELECT COUNT(*) FROM txn WHERE {sql}", params).fetchone()[0]
    rows = db.execute(f"SELECT date, time, category, amount, merchant, first, last, city, state, is_fraud "
                      f"FROM txn WHERE {sql} ORDER BY date DESC, time DESC LIMIT ?", params + [limit]).fetchall()
    return {"count": total, "rows": [dict(r) for r in rows], "where": sql, "params": params}


if __name__ == "__main__":
    _, n = build(sys.argv[1] if len(sys.argv) > 1 else HERE.parent.parent / ".vendor" / "sparkov" / "out")
    print(f"{n} transactions -> {DB}")
