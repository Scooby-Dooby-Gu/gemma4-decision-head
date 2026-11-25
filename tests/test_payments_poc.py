"""Payment search POC: query parsing, amount and date parsers, the SQL, and train/test separation."""

import sqlite3

from poc.payments.bench import head_labels, llm_filters, parse_queries, prefixes
from poc.payments.filters import QUESTIONS, amount_range, amounts, date_value, to_filters, where
from poc.payments.labels import cases_from, generate, leaks


def test_every_label_matches_the_parsers():
    for q in parse_queries():
        for text, exp in prefixes(q):
            if exp["amount"]:
                assert amount_range(exp["amount"], text) == exp["amount_range"], text
            if exp["date"]:
                assert date_value(exp["date"], text) == exp["date_value"], text
            for key, value in head_labels(exp).items():
                assert value in QUESTIONS[key]["criteria"], (text, key)


def test_prefix_expects_only_finished_segments():
    q = next(q for q in parse_queries() if q["text"] == "gas over $50")
    steps = prefixes(q)
    assert steps[0] == ("gas", {**steps[0][1], "category": "gas_transport"})
    assert steps[1][1]["amount"] is None  # "gas over": the amount isn't typed yet
    assert steps[2][1]["amount_range"] == [50, None]


def test_amounts_prefer_money():
    assert amounts("orders for 2 people over $50") == [50]
    assert amounts("more than 1k") == [1000]
    assert amount_range("between", "entre 50 y 100 dólares") == [50, 100]
    assert amount_range("around", "about 40 dollars") == [36, 44]
    assert amount_range("over", "over") is None


def test_dates():
    assert date_value("month", "hoteles en julio") == "2026-07-01..2026-07-31"
    assert date_value("weekday", "les vendredis") == "dow:4"
    assert date_value("last_month", "") == "2026-10-01..2026-10-31"
    assert date_value("past_week", "") == "2026-11-18..2026-11-24"
    assert date_value("month", "charges that") is None


def test_where_is_parameterised():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE txn (date TEXT, category TEXT, amount REAL, state TEXT, is_fraud INTEGER)")
    db.executemany("INSERT INTO txn VALUES (?,?,?,?,?)", [
        ("2026-11-20", "grocery_net", 30, "TX", 0),  # a Friday
        ("2026-11-21", "grocery_pos", 300, "TX", 1),
        ("2026-10-05", "travel", 900, "NY'; DROP TABLE txn; --", 0),
    ])
    count = lambda f: db.execute("SELECT COUNT(*) FROM txn WHERE " + where(f)[0], where(f)[1]).fetchone()[0]
    assert count({"category": "grocery", "channel": "online"}) == 1
    assert count({"date_value": "dow:4"}) == 1
    assert count({"amount_range": [100, None], "fraud": "fraud"}) == 1
    assert count({"date_value": "2026-10-01..2026-10-31"}) == 1
    assert count({"state": "NY'; DROP TABLE txn; --"}) == 1


def test_gate_waits_for_the_number():
    answers = {k: {"choice": next(iter(q["criteria"])), "confidence": 1.0, "abstain": False} for k, q in QUESTIONS.items()}
    answers["amount"] = {"choice": "over", "confidence": 0.9, "abstain": False}
    assert to_filters("purchases over", answers)["amount"] is None
    assert to_filters("purchases over $20", answers)["amount_range"] == [20, None]


def test_llm_filters():
    f = llm_filters({"category": "travel", "amount_min": 500, "weekdays": ["Saturday", "sunday"], "state": "TX"})
    assert f["amount_range"] == [500, None] and f["date_value"] == "dow:5,6" and f["state"] == "TX"
    assert llm_filters({"category": "misc", "state": "ZZ"})["category"] is None


def test_training_prefix_skips_the_phrase_being_typed():
    cases = cases_from([("over $50", {"amount": "over"}), ("in march", {"date": "month"})])
    assert cases[0]["skip"] == ["amount"]
    assert cases[2]["expected"]["amount"] == "over" and cases[2]["skip"] == ["date"]


def test_no_training_text_is_a_test_query():
    assert not leaks(generate())
