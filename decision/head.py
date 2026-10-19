"""The decision head: a linear readout of the backbone's last hidden state, one row per option.

Zero-data, the head is exactly Gemma's own next-token logits for the option handles:
    logit_k = logsumexp_m  softcap(U[handle_k, m] . h)
followed by content-free calibration (subtract alpha x the blank-state log-prior), averaging
over the views (listed and reversed order), a bias b and a temperature T.

Labelled cases upgrade the same head in place, so the zero-data model is the initialisation of
the trained one. `fit` picks the rung from the number of labels per option:
    rung 0  (< 5 per option)    nothing trained, zero-data head
    rung 1  (5 to 49)           b and T
    rung 2  (50 and up)         b, T and a per-option correction delta_k added to U[handle_k],
                                with weight decay pulling delta back to zero (that is, to Gemma)
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

ALPHA = 0.7  # share of the blank-state prior to divide out; Von found 1.0 over-corrects


class Head:
    def __init__(self, n_options, hidden_size, softcap=None, alpha=ALPHA):
        self.softcap = softcap
        self.params = {
            "b": mx.zeros((n_options,)),
            "log_t": mx.zeros(()),
            "delta": mx.zeros((n_options, hidden_size)),
            "alpha": mx.array(alpha),
        }
        self.rung = 0
        self.n_labels = 0

    # --- forward -----------------------------------------------------------------------

    def _view_logits(self, params, h, rows, mask):
        """h (N, D), rows (K, M, D), mask (K, M) -> raw option logits (N, K)."""
        w = rows + params["delta"][:, None, :]                    # (K, M, D)
        z = mx.einsum("nd,kmd->nkm", h, w)
        if self.softcap:
            z = self.softcap * mx.tanh(z / self.softcap)
        z = mx.where(mask[None], z, -1e9)
        return mx.logsumexp(z, axis=-1)

    def _logprobs(self, params, H, schema_arrays):
        """H: list over views of (N, D). Returns calibrated log-probs (N, K)."""
        total = 0
        for h, (rows, mask, prior) in zip(H, schema_arrays):
            z = self._view_logits(params, h, rows, mask)
            if prior is not None:
                z = z - params["alpha"] * prior[None]
            total = total + (z - mx.logsumexp(z, axis=-1, keepdims=True))
        avg = total / len(H)
        z = (avg + params["b"][None]) / mx.exp(params["log_t"])
        return z - mx.logsumexp(z, axis=-1, keepdims=True)

    @staticmethod
    def arrays(schema, with_prior=True):
        return [(mx.array(v.rows), mx.array(v.mask), mx.array(v.prior) if with_prior and v.prior is not None else None)
                for v in schema.views]

    def predict(self, schema, H, with_prior=True):
        """H: list over views of (N, D) numpy arrays -> probabilities (N, K) numpy."""
        lp = self._logprobs(self.params, [mx.array(h) for h in H], self.arrays(schema, with_prior))
        return np.array(mx.exp(lp))

    # --- training ----------------------------------------------------------------------

    def fit(self, schema, H, y, rung=None, steps=400, lr=0.05, weight_decay=1e-2):
        """H: list over views of (N, D) hidden states; y: (N,) option indices."""
        y = np.asarray(y)
        per_option = len(y) / len(schema.options)
        if rung is None:
            rung = 0 if per_option < 5 else 1 if per_option < 50 else 2
        self.rung, self.n_labels = rung, int(len(y))
        if rung == 0:
            return self
        trainable = ["b", "log_t"] + (["delta"] if rung >= 2 else [])
        Hm, ym, arrs = [mx.array(h) for h in H], mx.array(y), self.arrays(schema)
        frozen = {k: v for k, v in self.params.items() if k not in trainable}

        def loss_fn(train):
            lp = self._logprobs({**frozen, **train}, Hm, arrs)
            nll = -mx.take_along_axis(lp, ym[:, None], axis=1).mean()
            reg = 1e-3 * (train["b"] ** 2).sum()
            if "delta" in train:
                reg = reg + weight_decay * (train["delta"] ** 2).sum()
            return nll + reg

        train = {k: self.params[k] for k in trainable}
        opt = optim.Adam(learning_rate=lr if rung == 1 else lr / 10)
        step = mx.value_and_grad(loss_fn)
        for _ in range(steps):
            loss, grads = step(train)
            opt.update(train, grads)
            mx.eval(train, opt.state)
        self.params.update(train)
        return self

    # --- persistence -------------------------------------------------------------------

    def save(self, path, fingerprint):
        np.savez(path, fingerprint=fingerprint, rung=self.rung, n_labels=self.n_labels,
                 softcap=self.softcap if self.softcap else math.nan,
                 **{k: np.array(v) for k, v in self.params.items()})

    @classmethod
    def load(cls, path, fingerprint=None):
        d = np.load(path)
        if fingerprint is not None and str(d["fingerprint"]) != fingerprint:
            raise ValueError(f"{Path(path).name} was trained for a different version of this question")
        softcap = None if math.isnan(float(d["softcap"])) else float(d["softcap"])
        head = cls(*d["delta"].shape, softcap=softcap)
        head.params = {k: mx.array(d[k]) for k in ("b", "log_t", "delta", "alpha")}
        head.rung, head.n_labels = int(d["rung"]), int(d["n_labels"])
        return head
