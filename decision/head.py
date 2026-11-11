"""The decision head: a linear readout of the backbone's last hidden state, one row per option.

Zero-data, the head is exactly Gemma's own next-token logits for the option handles:
    logit_k = logsumexp_m  softcap(U[handle_k, m] . h)
followed by content-free calibration (subtract alpha x the blank-state log-prior), averaging
over the views (listed and reversed order), a bias b and a temperature T.

Labelled cases upgrade the same head in place, so the zero-data model is the initialisation of
the trained one. There are three rungs:
    rung 0   nothing trained, zero-data head
    rung 1   b and T
    rung 2   b, T and a per-option correction delta_k . h_hat added to each option's logit,
             where h_hat = h / |h| is the hidden state scaled to unit length. delta starts at
             zero (Gemma's own answer) and trains a fixed 400 steps. Weight decay is supported
             but off by default: on h_hat it held rung 2 to 83% on Banking77 g1 at 20 labels
             per option, against 94% without it.

The correction reads the unit-length hidden state, not h itself, because Gemma 4's final hidden
states are large (|h|_1 is about 5,000): reading h directly, one Adam step of 0.005 per weight
could move a logit by ~25 and training oscillated, sometimes ending mid-spike (accuracy 89% at
step 350, 54% at step 400). On h_hat one step moves a logit by at most lr x sqrt(D).

`train` is the upgrade path. The rung comes from the question type (rung 2 for `choice`,
rung 1 for `score` and `noul`), as measured on public datasets (see README, Experiments). It
holds out a fifth of each option's labels as a quiz and fits the rungs on the rest. The quiz
then does two jobs: it is a guard (a rung that scores clearly worse on it than a simpler rung
is replaced by that rung, once the quiz has two labels per option) and it sets the final
temperature, since rung 2 tends to be overconfident. The chosen rung is then refitted on all the labels. Early stopping on the quiz
was tried and dropped: held-out loss turns up after 10 to 40 steps while accuracy keeps
improving to ~400, so it stopped too soon (Banking77 g1, 20 labels per option: 92.3% vs 94.2%).
Picking the rung by quiz score alone was also dropped: with a few labels per option the quiz
is too small, and it picked no training in 4 of 5 draws on CLINC banking at 5 labels.
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

ALPHA = 0.7  # share of the blank-state prior to divide out; Von found 1.0 over-corrects
FORMAT = 2   # saved-head format; 2 = delta reads the unit-length hidden state

# Training defaults (see the README's Experiments section for how they were measured)
STEPS = 400
LR = {1: 0.05, 2: 0.02}
B_DECAY = 1e-3
DELTA_DECAY = 0.0
DEFAULT_RUNG = {"choice": 2, "score": 1, "noul": 1}
VAL_FRACTION = 0.2   # share of each option's labels held out as the quiz
GUARD = 0.05         # quiz loss (nats) by which a rung may trail a simpler one before it is dropped
GUARD_MIN_QUIZ = 2   # quiz labels needed for every option before the guard may drop a rung

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
        self.selection = None   # quiz results, when trained by `train`

    # --- forward -----------------------------------------------------------------------

    def _view_logits(self, params, h, rows, mask):
        """h (N, D), rows (K, M, D), mask (K, M) -> raw option logits (N, K)."""
        z = mx.einsum("nd,kmd->nkm", h, rows)
        if self.softcap:
            z = self.softcap * mx.tanh(z / self.softcap)
        z = mx.where(mask[None], z, -1e9)
        z = mx.logsumexp(z, axis=-1)
        h_hat = h / mx.linalg.norm(h, axis=-1, keepdims=True)
        return z + h_hat @ params["delta"].T

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

    def logprobs(self, schema, H, with_prior=True):
        """H: list over views of (N, D) numpy arrays -> log-probabilities (N, K) numpy."""
        return np.array(self._logprobs(self.params, [mx.array(h) for h in H], self.arrays(schema, with_prior)))

    def predict(self, schema, H, with_prior=True):
        """H: list over views of (N, D) numpy arrays -> probabilities (N, K) numpy."""
        return np.exp(self.logprobs(schema, H, with_prior))

    def nll(self, schema, H, y):
        """Mean negative log-likelihood of labels y."""
        lp = self.logprobs(schema, H)
        return float(-lp[np.arange(len(y)), np.asarray(y)].mean())

    # --- training ----------------------------------------------------------------------

    def fit(self, schema, H, y, rung=1, steps=STEPS, lr=None, weight_decay=DELTA_DECAY):
        """Train rung 1 or 2 on hidden states H (list over views of (N, D)) and labels y (N,)."""
        y = np.asarray(y)
        self.rung, self.n_labels = rung, int(len(y))
        if rung == 0:
            return self
        trainable = ["b", "log_t"] + (["delta"] if rung >= 2 else [])
        Hm, ym, arrs = [mx.array(h) for h in H], mx.array(y), self.arrays(schema)
        frozen = {k: v for k, v in self.params.items() if k not in trainable}

        def loss_fn(train):
            lp = self._logprobs({**frozen, **train}, Hm, arrs)
            nll = -mx.take_along_axis(lp, ym[:, None], axis=1).mean()
            reg = B_DECAY * (train["b"] ** 2).sum()
            if "delta" in train:
                reg = reg + weight_decay * (train["delta"] ** 2).sum()
            return nll + reg

        train = {k: self.params[k] for k in trainable}
        opt = optim.Adam(learning_rate=lr or LR[rung])
        step = mx.value_and_grad(loss_fn)
        for _ in range(steps):
            _, grads = step(train)
            opt.update(train, grads)
            mx.eval(train, opt.state)
        self.params.update(train)
        return self

    def calibrate(self, schema, H, y):
        """Rescale the temperature to minimise the loss on (H, y). Returns that loss (nats).

        Only the overall confidence changes, never which option is on top."""
        lp = self.logprobs(schema, H).astype(np.float64)
        y = np.asarray(y)
        best_loss, best_log = math.inf, 0.0
        for log_f in np.linspace(-1.5, 2.5, 81):  # temperature factor from 0.22x to 12x
            z = lp / math.exp(log_f)
            z = z - z.max(1, keepdims=True)
            z = z - np.log(np.exp(z).sum(1, keepdims=True))
            loss = float(-z[np.arange(len(y)), y].mean())
            if loss < best_loss:
                best_loss, best_log = loss, float(log_f)
        self.params["log_t"] = self.params["log_t"] + best_log
        return best_loss

    # --- persistence -------------------------------------------------------------------

    def save(self, path, fingerprint):
        np.savez(path, fingerprint=fingerprint, rung=self.rung, n_labels=self.n_labels, format=FORMAT,
                 softcap=self.softcap if self.softcap else math.nan,
                 **{k: np.array(v) for k, v in self.params.items()})

    @classmethod
    def load(cls, path, fingerprint=None):
        d = np.load(path)
        if fingerprint is not None and str(d["fingerprint"]) != fingerprint:
            raise ValueError(f"{Path(path).name} was trained for a different version of this question")
        fmt = int(d["format"]) if "format" in d.files else 1
        if fmt < FORMAT and int(d["rung"]) >= 2:
            raise ValueError(f"{Path(path).name} is a rung 2 head in an older format; retrain it")
        softcap = None if math.isnan(float(d["softcap"])) else float(d["softcap"])
        head = cls(*d["delta"].shape, softcap=softcap)
        head.params = {k: mx.array(d[k]) for k in ("b", "log_t", "delta", "alpha")}
        head.rung, head.n_labels = int(d["rung"]), int(d["n_labels"])
        return head


def split(y, fraction=VAL_FRACTION, seed=0):
    """Stratified (train_idx, val_idx): about `fraction` of each option's labels held out, at
    least one when the option has two or more, none when it has one."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for k in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == k))
        n_val = max(1, int(round(fraction * len(idx)))) if len(idx) >= 2 else 0
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.array(sorted(tr)), np.array(sorted(va))


def choose(quiz, rung):
    """The guard: keep `rung` unless its quiz loss trails the best simpler rung by more than
    GUARD; then try the next rung down. `quiz` maps rung -> calibrated quiz loss."""
    chosen = rung
    while chosen > 0 and quiz[chosen] > min(quiz[r] for r in range(chosen)) + GUARD:
        chosen -= 1
    return chosen


def train(schema, H, y, hidden_size, softcap, rung=None, seed=0):
    """The upgrade path: fit a head on labelled hidden states, guarded and calibrated on a quiz.

    `rung` defaults to DEFAULT_RUNG for the question type. A fifth of each option's labels is
    held out; every rung up to the requested one is fitted on the rest and scored on the quiz
    after calibrating its temperature there. The requested rung is kept unless it trails the
    best simpler rung by more than GUARD, in which case the next rung down is tried. The guard
    only acts when every option has GUARD_MIN_QUIZ quiz labels (about 10 labels per option):
    with one quiz label per option it fell back wrongly often enough to cost 1 to 2 points at
    5 labels per option, so below that the requested rung is kept and the quiz only calibrates.
    The kept rung is refitted on all labels and given the temperature correction its quiz run
    found.
    With under two labels per option on average there is no quiz, and the zero-data head is
    returned."""
    y = np.asarray(y)
    rung = DEFAULT_RUNG[schema.type] if rung is None else rung
    tr, va = split(y, seed=seed)
    new = lambda: Head(len(schema.options), hidden_size, softcap)
    if rung == 0 or len(va) < len(schema.options) or len(tr) == 0:
        head = new()
        head.n_labels = int(len(y))
        return head
    Htr, Hva = [h[tr] for h in H], [h[va] for h in H]
    guarded = int(np.bincount(y[va], minlength=len(schema.options))[np.unique(y)].min()) >= GUARD_MIN_QUIZ
    quiz, temp = {}, {}
    for r in (range(rung + 1) if guarded else [rung]):
        h = new().fit(schema, Htr, y[tr], rung=r)
        before = float(h.params["log_t"])
        quiz[r] = h.calibrate(schema, Hva, y[va])
        temp[r] = float(h.params["log_t"]) - before
    chosen = choose(quiz, rung) if guarded else rung
    final = new().fit(schema, H, y, rung=chosen) if chosen else new()
    final.params["log_t"] = final.params["log_t"] + temp[chosen]
    final.rung, final.n_labels = chosen, int(len(y))
    final.selection = {"requested": rung, "chosen": chosen, "guarded": guarded, "quiz_loss": quiz,
                       "quiz_size": int(len(va))}
    return final
