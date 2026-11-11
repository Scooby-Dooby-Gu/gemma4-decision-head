"""Tests on a tiny, randomly initialised and 4-bit quantized Gemma 4 text model.

They check the plumbing, not accuracy: that the zero-data head reproduces Gemma's own logits
exactly, that order debiasing and content-free calibration behave, and that the upgrade rungs
train and save. Run from the decision/ folder:  python -m pytest tests -q
"""

import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision import DecisionModel, build_schema  # noqa: E402
from decision.backbone import GemmaBackbone  # noqa: E402
from decision.head import Head  # noqa: E402
from decision.schema import ANSWER_PREFIX, SCORE_ANSWER_PREFIX  # noqa: E402


class WordTokenizer:
    """Word-level stand-in for Gemma's tokenizer: " Yes", "A", "0" are each one token."""

    def __init__(self, size):
        self.size, self.vocab = size, {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in re.findall(r"\s?\w+|\s?[^\w\s]|\s", text):
            if piece not in self.vocab:
                self.vocab[piece] = 1 + len(self.vocab) % (self.size - 1)
            ids.append(self.vocab[piece])
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return f"<user> {messages[0]['content']} <model> "


def tiny_gemma4(seed=0):
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import LanguageModel

    mx.random.seed(seed)
    cfg = TextConfig(hidden_size=64, num_hidden_layers=4, intermediate_size=128, num_attention_heads=2,
                     head_dim=32, global_head_dim=32, num_key_value_heads=1, num_kv_shared_layers=2,
                     vocab_size=2048, vocab_size_per_layer_input=2048, hidden_size_per_layer_input=32,
                     sliding_window=64, sliding_window_pattern=2)
    lm = LanguageModel(cfg)
    # Random init gives tiny logits; scale the tied embeddings up so softmaxes are not flat.
    lm.model.embed_tokens.weight = lm.model.embed_tokens.weight * 4
    nn.quantize(lm, group_size=32, bits=4)
    mx.eval(lm.parameters())
    return lm


@pytest.fixture(scope="module")
def backbone():
    return GemmaBackbone("tiny-random-gemma4", loaded=(tiny_gemma4(), WordTokenizer(2048)))


TRIAGE = {
    "department": {"type": "choice", "instructions": "Which team should handle this",
                   "criteria": {"billing": "Payment, invoice, refund", "technical": "Bugs, errors, outages",
                                "sales": "Pricing, plan upgrades"}},
    "frustration": {"type": "score", "instructions": "How frustrated the customer appears",
                    "criteria": ["Calm", "Frustrated but civil", "Very angry"]},
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
}


def test_handles_avoid_letters_used_in_the_schema():
    q = {"type": "choice", "instructions": "Pick a candidate",
         "criteria": {"cand_a": "Candidate A, the incumbent", "cand_b": "Candidate B, the challenger"}}
    s = build_schema("pick", q)
    used = {h for v in s.views for h in v.handles}
    assert not used & {"A", "B"}
    # Reversed view puts the same options under swapped letters.
    assert s.views[0].handles == s.views[1].handles[::-1]
    assert s.views[0].order == s.views[1].order[::-1]


def test_zero_data_head_equals_gemma_logits(backbone):
    dm = DecisionModel(backbone, debias_order=False)
    s = dm.schema("department", TRIAGE["department"])
    view = s.views[0]
    pre, tail = dm.prompt_ids(view, "I was charged twice")
    ids = pre + tail
    h, _ = backbone.encode(ids)
    full = backbone.logits(ids)
    head = Head(len(s.options), h.shape[-1], backbone.softcap)
    raw = np.array(head._view_logits(head.params, mx.array(h[None]), mx.array(view.rows), mx.array(view.mask)))[0]
    for k, hd in enumerate(view.handles):
        toks = backbone.single_token_ids(hd)
        expect = np.log(np.exp(full[toks]).sum())
        assert raw[k] == pytest.approx(expect, abs=2e-3)


def test_answers_have_jev_shape_and_valid_probabilities(backbone):
    dm = DecisionModel(backbone)
    ans = dm.answer("You charged my card TWICE. Fix this today!", TRIAGE)
    d = ans["department"]
    assert d["choice"] in TRIAGE["department"]["criteria"]
    assert sum(d["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert d["confidence"] == pytest.approx(max(d["probabilities"].values()), abs=1e-6)
    f = ans["frustration"]
    assert f["score"] in (0.0, 1.0, 2.0) and 0 <= f["expected"] <= 2
    assert 0 <= ans["is_urgent"]["noul"] <= 1
    assert all(isinstance(a["abstain"], bool) for a in ans.values())


def test_content_free_calibration_flattens_the_blank_state(backbone):
    """After dividing out alpha of the blank-state prior, the blank state itself sits closer to uniform."""
    dm = DecisionModel(backbone)
    s = dm.schema("department", TRIAGE["department"])
    H = [h[None] for h in dm.featurize(s, "N/A")]
    head = dm.heads[("department", s.fingerprint)]
    raw = head.predict(s, H, with_prior=False)[0]
    cal = head.predict(s, H)[0]
    uniform = np.full(len(s.options), 1 / len(s.options))
    assert np.abs(cal - uniform).sum() < np.abs(raw - uniform).sum()


def test_order_averaging_is_invariant_to_listing_order(backbone):
    """With both orders scored, reordering the criteria dict gives the same distribution."""
    dm = DecisionModel(backbone)
    q = TRIAGE["department"]
    flipped = dict(q, criteria=dict(reversed(list(q["criteria"].items()))))
    _, p1 = dm.probabilities("Stripe integration keeps failing", "department", q)
    _, p2 = dm.probabilities("Stripe integration keeps failing", "department", flipped)
    p2 = dict(zip(reversed(list(q["criteria"])), p2))
    for k, pk in zip(q["criteria"], p1):
        assert pk == pytest.approx(p2[k], abs=1e-4)


def synthetic_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    words = {"billing": "refund invoice charged card", "technical": "error crash bug outage",
             "sales": "pricing upgrade plan quote"}
    cases = []
    for i in range(n):
        label = list(words)[i % 3]
        toks = rng.choice(words[label].split(), size=4).tolist() + rng.choice(["please", "today", "hi"], 2).tolist()
        cases.append({"id": f"c{i}", "state": " ".join(rng.permutation(toks)), "expected": {"department": label}})
    return cases


def test_rungs_fit_their_training_data(backbone):
    dm = DecisionModel(backbone)
    q = TRIAGE["department"]
    H, y = dm.features(synthetic_cases(90), "department", q)
    s = dm.schema("department", q)
    zero = dm.heads[("department", s.fingerprint)]
    rung1 = Head(len(s.options), H[0].shape[-1], backbone.softcap).fit(s, H, y, rung=1)
    rung2 = Head(len(s.options), H[0].shape[-1], backbone.softcap).fit(s, H, y, rung=2)
    assert rung2.nll(s, H, y) < rung1.nll(s, H, y) <= zero.nll(s, H, y) + 1e-6


def test_correction_reads_the_unit_length_hidden_state():
    """Rung 2's term must not grow with |h|: Gemma 4's hidden states are large (|h|_1 ~ 5,000),
    and reading h directly made each optimizer step move logits by up to ~25."""
    rng = np.random.default_rng(0)
    head = Head(3, 8)
    head.params["delta"] = mx.array(rng.normal(size=(3, 8)).astype(np.float32))
    rows, mask = mx.array(rng.normal(size=(3, 1, 8)).astype(np.float32)), mx.ones((3, 1), dtype=mx.bool_)
    h = mx.array(rng.normal(size=(2, 8)).astype(np.float32))
    zero = dict(head.params, delta=mx.zeros((3, 8)))
    term = lambda x: np.array(head._view_logits(head.params, x, rows, mask) - head._view_logits(zero, x, rows, mask))
    np.testing.assert_allclose(term(h), term(h * 100), rtol=1e-4, atol=1e-4)


def test_calibrate_lowers_the_loss_without_changing_answers(backbone):
    dm = DecisionModel(backbone)
    q = TRIAGE["department"]
    H, y = dm.features(synthetic_cases(60), "department", q)
    s = dm.schema("department", q)
    head = Head(len(s.options), H[0].shape[-1], backbone.softcap).fit(s, [h[:30] for h in H], y[:30], rung=2)
    before, top = head.nll(s, [h[30:] for h in H], y[30:]), head.predict(s, H).argmax(1)
    after = head.calibrate(s, [h[30:] for h in H], y[30:])
    assert after <= before + 1e-6
    assert after == pytest.approx(head.nll(s, [h[30:] for h in H], y[30:]), abs=1e-4)
    np.testing.assert_array_equal(head.predict(s, H).argmax(1), top)


def test_guard_keeps_the_requested_rung_unless_a_simpler_one_is_clearly_better():
    from decision.head import GUARD, choose

    assert choose({0: 1.0, 1: 0.8, 2: 0.6}, 2) == 2
    assert choose({0: 1.0, 1: 0.8, 2: 0.8 + GUARD / 2}, 2) == 2  # within the margin: keep it
    assert choose({0: 1.0, 1: 0.8, 2: 0.8 + GUARD * 2}, 2) == 1
    assert choose({0: 0.5, 1: 0.8, 2: 0.9}, 2) == 0              # both trained rungs trail zero-data
    assert choose({0: 0.5, 1: 0.9}, 1) == 0


def test_train_picks_the_rung_for_the_question_type_and_saves_it(backbone, tmp_path):
    dm = DecisionModel(backbone)
    questions = {"department": TRIAGE["department"], "frustration": TRIAGE["frustration"]}
    cases = synthetic_cases(90)
    for i, c in enumerate(cases):
        c["expected"]["frustration"] = i % 3
    used = dm.fit(cases, questions)
    sel = {k: dm.heads[(k, dm.schema(k, q).fingerprint)].selection for k, q in questions.items()}
    assert sel["department"]["requested"] == 2 and sel["frustration"]["requested"] == 1
    assert used["department"] == sel["department"]["chosen"] and sel["department"]["quiz_size"] == 18
    assert set(sel["department"]["quiz_loss"]) == {0, 1, 2}

    dm.save_heads(tmp_path)
    fresh = DecisionModel(backbone)
    fresh.load_heads(tmp_path)
    H, _ = dm.features(cases, "department", questions["department"])
    s = fresh.schema("department", questions["department"])
    np.testing.assert_allclose(fresh.heads[("department", s.fingerprint)].predict(s, H),
                               dm.heads[("department", s.fingerprint)].predict(s, H), atol=1e-5)


def test_guard_waits_for_two_quiz_labels_per_option(backbone):
    """With one quiz label per option the guard misfired on real data, so it must not act; the
    requested rung is kept and the quiz only calibrates. From two per option it acts again."""
    dm = DecisionModel(backbone)
    q = {"department": TRIAGE["department"]}
    s = dm.schema("department", q["department"])
    for n_cases, guarded in [(15, False), (30, True)]:  # 5 and 10 labels per option
        cases = synthetic_cases(n_cases)
        dm.fit(cases, q)
        sel = dm.heads[("department", s.fingerprint)].selection
        assert sel["guarded"] is guarded
        assert set(sel["quiz_loss"]) == ({0, 1, 2} if guarded else {2})
        if not guarded:
            assert sel["chosen"] == sel["requested"] == 2


def test_too_few_labels_stay_zero_data(backbone):
    dm = DecisionModel(backbone)
    q = {"department": TRIAGE["department"]}
    cases = synthetic_cases(3)  # one per option: nothing to hold out
    assert dm.fit(cases, q) == {"department": 0}
    s = dm.schema("department", q["department"])
    assert dm.heads[("department", s.fingerprint)].selection is None


def test_old_format_rung2_heads_are_refused(tmp_path):
    head = Head(3, 8)
    head.rung = 2
    head.save(tmp_path / "x.npz", "fp")
    d = dict(np.load(tmp_path / "x.npz"))
    d.pop("format")
    np.savez(tmp_path / "old.npz", **d)
    with pytest.raises(ValueError, match="older format"):
        Head.load(tmp_path / "old.npz", "fp")
    assert Head.load(tmp_path / "x.npz", "fp").rung == 2


def test_trained_head_is_ignored_when_the_question_changes(backbone, tmp_path):
    dm = DecisionModel(backbone)
    q = {"department": TRIAGE["department"]}
    cases = synthetic_cases(60)
    dm.fit(cases, q)
    dm.save_heads(tmp_path)
    edited = dict(TRIAGE["department"], instructions="Which team owns this ticket")
    fresh = DecisionModel(backbone)
    fresh.load_heads(tmp_path)
    s = fresh.schema("department", edited)
    assert fresh.heads[("department", s.fingerprint)].rung == 0


def test_score_prefix_ends_with_a_space_so_the_digit_is_the_next_token():
    """Gemma writes " 1" as a space then a digit, so after a bare "Answer:" it wants a space and the
    digits get almost no probability (0.3% on the real model). Score views must end on the space."""
    views = {k: build_schema(k, q).views for k, q in TRIAGE.items()}
    assert all(v.answer_prefix == SCORE_ANSWER_PREFIX == ANSWER_PREFIX + " " for v in views["frustration"])
    # Letters and Yes/No are single " B" / " Yes" tokens, so they keep the bare prefix.
    assert all(v.answer_prefix == ANSWER_PREFIX for v in views["department"] + views["is_urgent"])


def test_handle_mass_is_the_softmax_share_of_the_handle_tokens(backbone):
    dm = DecisionModel(backbone, debias_order=False)
    s = dm.schema("frustration", TRIAGE["frustration"])
    view = s.views[0]
    (mass,) = dm.handle_mass(s, "Why is this still broken")
    z = backbone.logits(sum(dm.prompt_ids(view, "Why is this still broken"), []))
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    toks = sorted({t for hd in view.handles for t in backbone.single_token_ids(hd)})
    assert 0 <= mass <= 1 and mass == pytest.approx(p[toks].sum(), abs=1e-5)


def test_fingerprint_changes_with_the_prompt_setup():
    """A head is fitted to the prompt template and the views, so changing either must not reuse it."""
    import decision.schema as schema

    q = TRIAGE["department"]
    both, listed = build_schema("d", q).fingerprint, build_schema("d", q, debias_order=False).fingerprint
    assert both != listed
    old = schema.TEMPLATE_VERSION
    try:
        schema.TEMPLATE_VERSION = old + 1
        assert build_schema("d", q).fingerprint != both
    finally:
        schema.TEMPLATE_VERSION = old


LONG_STATE = " ".join(["The invoice from last month still shows the old plan and"] * 12) + " please fix it today"


@pytest.mark.parametrize("state_first", [False, True])
@pytest.mark.parametrize("state", ["Charged twice, fix it!", LONG_STATE])
def test_cached_prefix_gives_the_same_hidden_states_as_full_passes(backbone, state, state_first):
    """A cached prefix plus the rest must equal a full pass per view, in both layouts. LONG_STATE is
    longer than the tiny model's 64-token sliding window, so the rotating cache wraps too."""
    cached = DecisionModel(backbone, state_first=state_first)
    full = DecisionModel(backbone, state_first=state_first, use_cache=False)
    assert len(backbone.tokenize_chat(LONG_STATE)) > 64
    sc = {k: cached.schema(k, q) for k, q in TRIAGE.items()}
    sf = {k: full.schema(k, q) for k, q in TRIAGE.items()}
    a, b = cached.featurize_all(state, sc), full.featurize_all(state, sf)
    for key in TRIAGE:
        assert len(a[key]) == len(sc[key].views)
        for ha, hb in zip(a[key], b[key]):
            np.testing.assert_allclose(ha, hb, atol=2e-3, rtol=2e-3)
    # The shared prefix is copied, not modified: answering again gives the same result.
    again = cached.featurize_all(state, sc)
    for key in TRIAGE:
        for ha, hc in zip(a[key], again[key]):
            np.testing.assert_array_equal(ha, hc)
    assert cached.answer(state, TRIAGE).keys() == TRIAGE.keys()


def test_state_first_shares_one_prefix_across_questions(backbone):
    dm = DecisionModel(backbone, state_first=True)
    prefixes = {tuple(dm.prompt_ids(v, "Refund please")[0])
                for k, q in TRIAGE.items() for v in dm.schema(k, q).views}
    assert len(prefixes) == 1 and len(prefixes.pop()) > 0


def test_question_first_shares_each_view_prefix_across_cases(backbone):
    dm = DecisionModel(backbone)
    for k, q in TRIAGE.items():
        for v in dm.schema(k, q).views:
            a, b = dm.prompt_ids(v, "Refund please"), dm.prompt_ids(v, "The app crashes on login")
            assert a[0] == b[0] and a[1] != b[1]


def test_default_prompts_are_the_original_question_first_ones(backbone):
    """The default layout is the original one; state_first is opt-in and changes the fingerprint."""
    q = TRIAGE["is_urgent"]
    (v,) = build_schema("u", q).views
    assert v.prompt("Hi") == ("Statement: The message conveys urgency.\n\nInput:\nHi\n\n"
                              "Is the statement true for this input? Reply with only Yes or No.")
    # The cached split is the same text. (On Gemma's real tokenizer the two pieces also tokenize
    # exactly like the whole prompt, checked on all 186 sample-suite prompts; this test's word
    # tokenizer glues newlines to the next word, so token ids are not compared here.)
    assert "".join(v.split("Hi")) == v.prompt("Hi") and v.split("Hi")[0].endswith("Input:\n")
    (w,) = build_schema("u", q, state_first=True).views
    assert w.prompt("Hi").startswith("Input:\nHi\n\nStatement:")
    assert build_schema("u", q).fingerprint != build_schema("u", q, state_first=True).fingerprint
