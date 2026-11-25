"""Search POC pieces that don't need the model: query parsing, prefixes, age numbers, gating."""

from poc.search.filters import QUESTIONS, age_range, numbers, to_filters
from poc.search.measure import HERE, parse_queries, prefixes


def answer(choice, confidence=0.9, abstain=False):
    return {"choice": choice, "confidence": confidence, "abstain": abstain}


NOTHING = {"state_east": answer("none"), "state_west": answer("none"), "kids": answer("not_mentioned"),
           "age": answer("not_mentioned"), "marital": answer("not_mentioned")}


def test_every_query_parses_and_choice_questions_fit_the_handles():
    queries = parse_queries(HERE / "queries.txt")
    assert len(queries) > 100
    assert {q["lang"] for q in queries} == {"en", "es", "fr"}
    for q in QUESTIONS.values():
        assert len(q["criteria"]) <= 34


def test_prefixes_expect_a_filter_only_once_its_segment_is_typed():
    q = {"segments": [("new york", {"state": "NY"}), ("with kids", {"kids": "has_kids"})]}
    steps = prefixes(q)
    assert [t for t, _ in steps] == ["new", "new york", "new york with", "new york with kids"]
    assert steps[0][1]["state"] is None and steps[1][1]["state"] == "NY"
    assert steps[2][1]["kids"] is None and steps[3][1]["kids"] == "has_kids"


def test_age_numbers_in_three_languages():
    assert numbers("over 40") == [40]
    assert numbers("in their forties") == [40]
    assert numbers("en sus cuarentas") == [40]
    assert numbers("la quarantaine") == [40]
    assert numbers("in their 40s") == [40]


def test_age_ranges():
    assert age_range("older_than", "over 40") == [41, None]
    assert age_range("older_than", "50+") == [50, None]
    assert age_range("younger_than", "under 30") == [None, 29]
    assert age_range("between", "between 45 and 30") == [30, 45]
    assert age_range("decade", "in their 40s") == [40, 49]
    assert age_range("seniors", "retirees") == [65, None]
    assert age_range("older_than", "over") is None  # number not typed yet


def test_filters_wait_for_confidence_and_the_number():
    a = dict(NOTHING, state_east=answer("NY"), age=answer("older_than"))
    f = to_filters("new york over", a)
    assert f["state"] == "NY" and f["age"] is None  # "over" with no number stays off
    f = to_filters("new york over 40", a)
    assert f["age"] == "older_than" and f["age_range"] == [41, None]
    a = dict(NOTHING, state_east=answer("NJ", confidence=0.4))
    assert to_filters("new", a)["state"] is None
    a = dict(NOTHING, state_east=answer("NJ", abstain=True))
    assert to_filters("new", a, threshold=0.0)["state"] is None


def test_crm_query_is_parameterised():
    from poc.search.crm import query

    where, params = query({"state": "NY", "kids": "has_kids", "age_range": [41, None], "marital": None})
    assert where == "state = ? AND kids > 0 AND age >= ?" and params == ["NY", 41]
    assert query({}) == ("1", [])


def test_training_prefixes_leave_the_phrase_being_typed_unlabelled():
    from poc.search.labels import cases_from

    cases = cases_from([("clients", {}), ("married couples", {"marital": "married"}), ("in ohio", {"state_east": "OH"})])
    by_text = {c["state"]: c for c in cases}
    assert by_text["clients married"]["skip"] == ["marital"]  # "married" may already mean married
    assert by_text["clients married couples"]["expected"]["marital"] == "married"
    assert by_text["clients married couples in"]["skip"] == ["state_east"]
    assert by_text["clients married couples in"]["expected"]["marital"] == "married"
    assert by_text["clients married couples in ohio"]["final"]


def test_training_cases_never_repeat_a_test_query():
    import json

    from poc.search.labels import OUT, leaks

    assert leaks(json.loads(OUT.read_text())) == []
