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
from decision.schema import ANSWER_PREFIX  # noqa: E402


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
    ids = backbone.tokenize_chat(view.prompt("I was charged twice"), ANSWER_PREFIX)
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


def test_upgrade_rungs_train_and_improve_fit(backbone, tmp_path):
    dm = DecisionModel(backbone)
    q = {"department": TRIAGE["department"]}
    cases = synthetic_cases(180)
    H, y = dm.features(cases, "department", q["department"])
    s = dm.schema("department", q["department"])

    def nll(head):
        p = head.predict(s, H)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    zero = dm.heads[("department", s.fingerprint)]
    assert dm.fit(cases[:6], q, cached={"department": ([h[:6] for h in H], y[:6])}) == {"department": 0}
    assert dm.fit(cases[:30], q, cached={"department": ([h[:30] for h in H], y[:30])}) == {"department": 1}
    rung1 = dm.heads[("department", s.fingerprint)]
    assert dm.fit(cases, q, cached={"department": (H, y)}) == {"department": 2}
    rung2 = dm.heads[("department", s.fingerprint)]
    assert nll(rung2) < nll(rung1) <= nll(zero) + 1e-6

    dm.save_heads(tmp_path)
    fresh = DecisionModel(backbone)
    fresh.load_heads(tmp_path)
    fresh.schema("department", q["department"])
    loaded = fresh.heads[("department", s.fingerprint)]
    assert loaded.rung == 2
    np.testing.assert_allclose(loaded.predict(s, H), rung2.predict(s, H), atol=1e-5)


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
