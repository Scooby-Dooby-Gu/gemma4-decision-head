"""Turn one Jev-style question into the prompts ("views") the backbone reads.

A question is the same dict jev-vs-local's cases/*.json use:
  {"type": "choice", "instructions": ..., "criteria": {key: description, ...}}
  {"type": "score",  "instructions": ..., "criteria": [level 0 text, level 1 text, ...]}
  {"type": "noul",   "instructions": ...}

Each option is shown under a short handle (a letter, digit, Yes or No) and the answer is read
from the logit of that handle token. Two defaults from the design doc's "Inherent biases":
  - choice handles are letters that appear nowhere in the schema as a standalone token, so a
    handle never collides with an option called "Candidate B" or a sentence starting "A ...";
  - choice questions are scored twice, in the listed and in the reversed order, and the
    per-option log-probabilities are averaged, which cancels most position and letter bias.

Two prompt layouts. Each prompt is split into a shared part, whose KV cache the backbone
computes once and reuses, and the rest, which is run per call:
  - question first (default): [Task ... options ... Input:] | <state> ... Reply with ...
    The shared part is the question, the same for every case, so each case only runs its state
    and the closing line. This is the original prompt, so answers are unchanged.
  - state first (opt-in, `state_first=True`): [Input: <state>] | Task ... options ... Reply with ...
    The shared part is the state, the same for every question about one case, so the state is
    read once per case. This pays off for long states (about 2x faster at ~300 state tokens,
    2.7x at ~1,300), but on the sample suites Gemma 4 E4B answers a little less accurately with
    the question after the input (is_urgent 83% vs 92%), so re-check accuracy on your cases.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no "I": it is too common as a word to be a clean handle
DIGITS = "123456789"
ANSWER_PREFIX = "Answer:"
# Gemma writes " 1" as two tokens (a space, then the digit), so after "Answer:" its next token is
# a bare space and the digits get almost no probability. Score views end the prefix with the
# space, so the digit is the very next token. Letters and Yes/No have " B" / " Yes" tokens and
# keep the plain prefix.
SCORE_ANSWER_PREFIX = ANSWER_PREFIX + " "
BLANK_STATE = "N/A"
# Bump whenever prompt wording or prefixes change: it is part of the fingerprint, so heads
# trained on the old prompts are not reused.
TEMPLATE_VERSION = 2


@dataclass
class View:
    order: list            # option indices in the order shown
    handles: list          # handle shown for option index k (indexed by option, not position)
    task: str              # the question and its options
    reply: str             # the closing instruction, worded for where the state sits
    state_first: bool = False
    answer_prefix: str = ANSWER_PREFIX  # start of the model's turn; the handle is the next token
    rows: np.ndarray = None   # (K, M, D) unembedding rows of each option's handle tokens
    mask: np.ndarray = None   # (K, M) which of the M slots are real tokens
    prior: np.ndarray = None  # (K,) zero-data log-probs on a blank state (content-free prior)

    def split(self, state):
        """(shared, rest): the cacheable part of the prompt and the part run per call.

        Question first, the question is shared by every case; state first, the state is shared by
        every question. Either way prompt = shared + rest."""
        if self.state_first:
            return f"Input:\n{state}\n\n", f"{self.task}\n\n{self.reply}"
        return f"{self.task}\n\nInput:\n", f"{state}\n\n{self.reply}"

    def prompt(self, state):
        return "".join(self.split(state))


@dataclass
class Schema:
    key: str
    question: dict
    options: list          # answer values: criteria keys, level strings, or ["true", "false"]
    views: list = field(default_factory=list)

    @property
    def type(self):
        return self.question["type"]

    @property
    def fingerprint(self):
        # Option order matters (it changes the prompt), and so do the prompt template and the
        # number of views (order debiasing on or off), since a trained head is fitted to both.
        layout = "state_first" if self.views and self.views[0].state_first else "question_first"
        blob = json.dumps([TEMPLATE_VERSION, layout, len(self.views), self.question]).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def label_index(self, expected):
        """Map a case's expected answer (as jev-vs-local writes it) to an option index."""
        if self.type == "choice":
            return self.options.index(expected)
        if self.type == "score":
            return int(expected)
        return 0 if expected else 1


def _standalone(text, token):
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text) is not None


def pick_handles(n, schema_text, usable):
    """First n letters (then digits) that never appear standalone in the schema and are one token."""
    picked = [c for c in LETTERS + DIGITS if not _standalone(schema_text, c) and usable(c)]
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} clean single-token handles for {n} options")
    return picked[:n]


def build_schema(key, question, usable=lambda handle: True, debias_order=True, state_first=False):
    where = "for the input above" if state_first else ""
    qtype = question["type"]
    instr = question["instructions"].strip().rstrip(".")
    if qtype == "choice":
        options = list(question["criteria"])
        text = instr + "\n" + "\n".join(f"{k} {v}" for k, v in question["criteria"].items())
        letters = pick_handles(len(options), text, usable)
        orders = [list(range(len(options)))]
        if debias_order and len(options) > 1:
            orders.append(orders[0][::-1])
        views = []
        for order in orders:
            handles = [None] * len(options)
            lines = []
            for pos, k in enumerate(order):
                handles[k] = letters[pos]
                lines.append(f"{letters[pos]}. {options[k]}: {question['criteria'][options[k]]}")
            task = f"Task: {instr}.\n\nOptions:\n" + "\n".join(lines)
            reply = f"Reply with only the letter of the best option{' ' + where if where else ''}."
            views.append(View(order, handles, task, reply, state_first))
        return Schema(key, question, options, views)
    if qtype == "score":
        levels = question["criteria"]
        options = [str(i) for i in range(len(levels))]
        lines = [f"{i}. {c}" for i, c in enumerate(levels)]
        task = f"Task: rate {instr[0].lower() + instr[1:]}.\n\nLevels:\n" + "\n".join(lines)
        reply = f"Reply with only the level number{' ' + where if where else ''}."
        view = View(list(range(len(levels))), options[:], task, reply, state_first,
                    answer_prefix=SCORE_ANSWER_PREFIX)
        return Schema(key, question, options, [view])
    if qtype == "noul":
        task = f"Statement: {instr}."
        reply = f"Is the statement true for {'the input above' if state_first else 'this input'}? Reply with only Yes or No."
        return Schema(key, question, ["true", "false"], [View([0, 1], ["Yes", "No"], task, reply, state_first)])
    raise ValueError(f"unknown question type {qtype!r}")
