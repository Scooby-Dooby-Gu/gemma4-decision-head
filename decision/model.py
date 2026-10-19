"""DecisionModel: Jev-style questions in, Jev-style answers with probabilities out, no decoding.

    from decision import DecisionModel
    dm = DecisionModel()                                   # loads Gemma 4 E4B once
    dm.answer("Charged twice this month!", questions)      # works with zero training data
    dm.fit(cases, questions)                               # optional upgrade from labelled cases
"""

from pathlib import Path

import numpy as np

from .head import Head
from .schema import ANSWER_PREFIX, BLANK_STATE, build_schema

ABSTAIN_MARGIN = 0.10  # top-1 minus top-2 probability below this sets "abstain": true


class DecisionModel:
    def __init__(self, backbone=None, *, debias_order=True, abstain_margin=ABSTAIN_MARGIN):
        if backbone is None:
            from .backbone import GemmaBackbone

            backbone = GemmaBackbone()
        self.backbone = backbone
        self.debias_order = debias_order
        self.abstain_margin = abstain_margin
        self.schemas = {}  # (key, fingerprint) -> Schema with rows and priors filled in
        self.heads = {}    # (key, fingerprint) -> Head

    # --- schema compilation (once per question, cached) --------------------------------

    def schema(self, key, question):
        probe = build_schema(key, question, debias_order=self.debias_order)
        cache_key = (key, probe.fingerprint)
        if cache_key in self.schemas:
            return self.schemas[cache_key]
        bb = self.backbone
        s = build_schema(key, question, usable=lambda c: bool(bb.single_token_ids(c)),
                         debias_order=self.debias_order)
        for view in s.views:
            token_sets = [bb.single_token_ids(hd) for hd in view.handles]
            if not all(token_sets):
                bad = [hd for hd, t in zip(view.handles, token_sets) if not t]
                raise ValueError(f"{key}: handles {bad} are not single tokens for this tokenizer")
            m = max(len(t) for t in token_sets)
            rows, mask = [], np.zeros((len(token_sets), m), dtype=bool)
            for k, toks in enumerate(token_sets):
                r = bb.unembed(toks)
                rows.append(np.pad(r, ((0, m - len(toks)), (0, 0))))
                mask[k, : len(toks)] = True
            view.rows, view.mask = np.stack(rows), mask
        self.schemas[cache_key] = s
        head = self.heads.get(cache_key) or Head(len(s.options), s.views[0].rows.shape[-1], bb.softcap)
        self.heads[cache_key] = head
        # Content-free prior: the zero-data head's answer on a blank state, one per view.
        blank = self.featurize(s, BLANK_STATE)
        base = Head(len(s.options), s.views[0].rows.shape[-1], bb.softcap)
        for view, h in zip(s.views, blank):
            single = type(s)(s.key, s.question, s.options, [view])
            view.prior = np.log(base.predict(single, [h[None]], with_prior=False)[0] + 1e-12)
        return s

    # --- backbone passes ---------------------------------------------------------------

    def featurize(self, schema, state):
        """One forward pass per view; returns a list over views of the last hidden state (D,)."""
        out = []
        for view in schema.views:
            ids = self.backbone.tokenize_chat(view.prompt(state), ANSWER_PREFIX)
            last, _ = self.backbone.encode(ids)
            out.append(last)
        return out

    def embed(self, text):
        """Mean-pooled final hidden state of `text` (for similarity or shortlist uses, not the head)."""
        ids = self.backbone.tokenize_chat(text)
        return self.backbone.encode(ids)[1]

    # --- answering ---------------------------------------------------------------------

    def probabilities(self, state, key, question):
        s = self.schema(key, question)
        H = [h[None] for h in self.featurize(s, state)]
        return s, self.heads[(key, s.fingerprint)].predict(s, H)[0]

    def answer_one(self, state, key, question):
        s, p = self.probabilities(state, key, question)
        top = np.argsort(-p)
        confidence = float(p[top[0]])
        margin = confidence - float(p[top[1]]) if len(p) > 1 else 1.0
        if s.type == "noul":
            return {"type": "noul", "noul": float(p[0]), "confidence": confidence,
                    "abstain": margin < self.abstain_margin}
        probs = {o: float(pk) for o, pk in zip(s.options, p)}
        out = {"type": s.type, "confidence": confidence, "probabilities": probs,
               "abstain": margin < self.abstain_margin}
        if s.type == "choice":
            out["choice"] = s.options[top[0]]
        else:
            out["score"] = float(top[0])
            out["expected"] = float(np.dot(np.arange(len(p)), p))
        return out

    def answer(self, state, questions):
        """Jev /v1/systemone `answers` for one state."""
        return {key: self.answer_one(state, key, q) for key, q in questions.items()}

    # --- upgrade path ------------------------------------------------------------------

    def features(self, cases, key, question):
        """Hidden states for labelled cases: (list over views of (N, D), labels (N,))."""
        s = self.schema(key, question)
        feats = [self.featurize(s, c["state"]) for c in cases]
        H = [np.stack([f[v] for f in feats]) for v in range(len(s.views))]
        y = np.array([s.label_index(c["expected"][key]) for c in cases])
        return H, y

    def fit(self, cases, questions, rung=None, cached=None):
        """Train each question's head on labelled cases (jev-vs-local format). Returns rungs used.

        `cached` can map question key -> (H, y) from `features` to skip the backbone passes."""
        used = {}
        for key, q in questions.items():
            s = self.schema(key, q)
            H, y = (cached or {}).get(key) or self.features(cases, key, q)
            head = Head(len(s.options), H[0].shape[-1], self.backbone.softcap)
            self.heads[(key, s.fingerprint)] = head.fit(s, H, y, rung=rung)
            used[key] = head.rung
        return used

    def save_heads(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for (key, fp), head in self.heads.items():
            if head.rung > 0:
                head.save(folder / f"{key}-{fp}.npz", fp)

    def load_heads(self, folder):
        """Load trained heads; questions without one stay zero-data."""
        for path in Path(folder).glob("*.npz"):
            key, fp = path.stem.rsplit("-", 1)
            self.heads[(key, fp)] = Head.load(path, fp)
