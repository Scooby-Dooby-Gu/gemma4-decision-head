"""DecisionModel: Jev-style questions in, Jev-style answers with probabilities out, no decoding.

    from decision import DecisionModel
    dm = DecisionModel()                                   # loads Gemma 4 E4B once
    dm.answer("Charged twice this month!", questions)      # works with zero training data
    dm.fit(cases, questions)                               # optional upgrade from labelled cases
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .head import Head, train
from .schema import BLANK_STATE, build_schema

ABSTAIN_MARGIN = 0.10  # top-1 minus top-2 probability below this sets "abstain": true
PREFIX_CACHE_SIZE = 64  # prompt prefixes whose KV cache is kept (least recently used dropped)


class DecisionModel:
    def __init__(self, backbone=None, *, debias_order=True, abstain_margin=ABSTAIN_MARGIN,
                 state_first=False, use_cache=True):
        if backbone is None:
            from .backbone import GemmaBackbone

            backbone = GemmaBackbone()
        self.backbone = backbone
        self.debias_order = debias_order
        self.abstain_margin = abstain_margin
        # Prompt layout (see schema.py): question first (default) caches each question's prefix
        # across cases; state_first caches each case's state across its questions.
        self.state_first = state_first
        self.use_cache = use_cache  # False runs a full pass per view (same answers, slower)
        self._prefixes = OrderedDict()  # tuple(prefix ids) -> KV cache from backbone.prefill
        self.schemas = {}  # (key, fingerprint) -> Schema with rows and priors filled in
        self.heads = {}    # (key, fingerprint) -> Head

    # --- schema compilation (once per question, cached) --------------------------------

    def schema(self, key, question):
        probe = build_schema(key, question, debias_order=self.debias_order, state_first=self.state_first)
        cache_key = (key, probe.fingerprint)
        if cache_key in self.schemas:
            return self.schemas[cache_key]
        bb = self.backbone
        s = build_schema(key, question, usable=lambda c: bool(bb.single_token_ids(c)),
                         debias_order=self.debias_order, state_first=self.state_first)
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
        # Not with state_first: there Gemma reads the blank as a real input and answers it almost
        # certainly ("N/A" is level 0, not urgent), and dividing that out over-corrects badly
        # (frustration 75% -> 33% on the sample suite). Five blank variants and alpha 0.3 did not
        # fix it, so state-first prompts run without the prior.
        if self.state_first:
            return s
        blank = self.featurize(s, BLANK_STATE)
        base = Head(len(s.options), s.views[0].rows.shape[-1], bb.softcap)
        for view, h in zip(s.views, blank):
            single = type(s)(s.key, s.question, s.options, [view])
            view.prior = np.log(base.predict(single, [h[None]], with_prior=False)[0] + 1e-12)
        return s

    # --- backbone passes ---------------------------------------------------------------

    def prompt_ids(self, view, state):
        """(prefix_ids, tail_ids): the cacheable shared part and the per-call rest of the prompt."""
        shared, rest = view.split(state)
        return self.backbone.tokenize_split(shared, rest, view.answer_prefix)

    def _prefix(self, ids):
        """KV cache for prompt prefix `ids`, computed once and kept (LRU, PREFIX_CACHE_SIZE)."""
        key = tuple(ids)
        if key in self._prefixes:
            self._prefixes.move_to_end(key)
        else:
            self._prefixes[key] = self.backbone.prefill(ids)
            if len(self._prefixes) > PREFIX_CACHE_SIZE:
                self._prefixes.popitem(last=False)
        return self._prefixes[key]

    def featurize_all(self, state, schemas):
        """Hidden states for several questions about one state: {key: list over views of (D,)}.

        Each view runs only the part after its cached prefix: the state and closing line (question
        first), or the question (state first)."""
        out = {}
        for key, schema in schemas.items():
            out[key] = []
            for view in schema.views:
                pre, tail = self.prompt_ids(view, state)
                if self.use_cache:
                    last, _ = self.backbone.encode(tail, prefix=self._prefix(pre))
                else:
                    last, _ = self.backbone.encode(pre + tail)
                out[key].append(last)
        return out

    def featurize(self, schema, state):
        """Hidden states for one question: a list over views of the last hidden state (D,)."""
        return self.featurize_all(state, {schema.key: schema})[schema.key]

    def handle_mass(self, schema, state):
        """Share of Gemma's full next-token probability that lands on the answer handles, per view.

        The head only compares the handles with each other, so it gives confident-looking answers
        even when Gemma wants to write something else entirely. A low share (well under 0.5) means
        the prompt does not lead Gemma to answer with a handle, and the probabilities are noise.
        Costs a full-vocabulary pass per view, so it is a diagnostic, not part of answering."""
        out = []
        for view in schema.views:
            pre, tail = self.prompt_ids(view, state)
            z = self.backbone.logits(pre + tail)
            p = np.exp(z - z.max())
            toks = sorted({t for hd in view.handles for t in self.backbone.single_token_ids(hd)})
            out.append(float(p[toks].sum() / p.sum()))
        return out

    def embed(self, text):
        """Mean-pooled final hidden state of `text` (for similarity or shortlist uses, not the head)."""
        ids = self.backbone.tokenize_chat(text)
        return self.backbone.encode(ids)[1]

    # --- answering ---------------------------------------------------------------------

    def probabilities(self, state, key, question):
        s = self.schema(key, question)
        return s, self._predict(s, self.featurize(s, state))

    def _predict(self, s, feats):
        return self.heads[(s.key, s.fingerprint)].predict(s, [h[None] for h in feats])[0]

    def answer_one(self, state, key, question):
        s, p = self.probabilities(state, key, question)
        return self._format(s, p)

    def _format(self, s, p):
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
        """Jev /v1/systemone `answers` for one state (the state is read once for all questions)."""
        schemas = {key: self.schema(key, q) for key, q in questions.items()}
        feats = self.featurize_all(state, schemas)
        return {key: self._format(s, self._predict(s, feats[key])) for key, s in schemas.items()}

    # --- upgrade path ------------------------------------------------------------------

    def features(self, cases, key, question):
        """Hidden states for labelled cases: (list over views of (N, D), labels (N,))."""
        return self.features_all(cases, {key: question})[key]

    def features_all(self, cases, questions):
        """{key: (H, y)} for every question, reading each case's state once."""
        schemas = {key: self.schema(key, q) for key, q in questions.items()}
        feats = [self.featurize_all(c["state"], schemas) for c in cases]
        out = {}
        for key, s in schemas.items():
            H = [np.stack([f[key][v] for f in feats]) for v in range(len(s.views))]
            y = np.array([s.label_index(c["expected"][key]) for c in cases])
            out[key] = (H, y)
        return out

    def fit(self, cases, questions, rung=None, cached=None):
        """Train each question's head on labelled cases (jev-vs-local format). Returns rungs used.

        Each head goes through `head.train`: the rung defaults to the question type's (rung 2
        for choice, rung 1 otherwise), a held-out fifth of the labels guards against a rung that
        does worse than a simpler one and calibrates the confidence, and the head is refitted on
        all labels. `rung` forces the highest rung tried. `cached` can map question key ->
        (H, y) from `features` to skip the backbone passes."""
        used = {}
        cached = dict(cached or {})
        missing = {k: q for k, q in questions.items() if k not in cached}
        if missing:
            cached.update(self.features_all(cases, missing))
        for key, q in questions.items():
            s = self.schema(key, q)
            H, y = cached[key]
            head = train(s, H, y, H[0].shape[-1], self.backbone.softcap, rung=rung)
            self.heads[(key, s.fingerprint)] = head
            used[key] = head.rung
        return used

    def save_heads(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for (key, fp), head in self.heads.items():
            if head.rung > 0 or head.selection is not None:  # trained, if only its temperature
                head.save(folder / f"{key}-{fp}.npz", fp)

    def load_heads(self, folder):
        """Load trained heads; questions without one stay zero-data."""
        for path in Path(folder).glob("*.npz"):
            key, fp = path.stem.rsplit("-", 1)
            self.heads[(key, fp)] = Head.load(path, fp)
