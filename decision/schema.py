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
    head: str              # prompt text before the state
    tail: str              # prompt text after the state
    answer_prefix: str = ANSWER_PREFIX  # start of the model's turn; the handle is the next token
    rows: np.ndarray = None   # (K, M, D) unembedding rows of each option's handle tokens
    mask: np.ndarray = None   # (K, M) which of the M slots are real tokens
    prior: np.ndarray = None  # (K,) zero-data log-probs on a blank state (content-free prior)

    def prompt(self, state):
        return self.head + state + self.tail


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
        blob = json.dumps([TEMPLATE_VERSION, len(self.views), self.question]).encode()
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


def build_schema(key, question, usable=lambda handle: True, debias_order=True):
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
            head = f"Task: {instr}.\n\nOptions:\n" + "\n".join(lines) + "\n\nInput:\n"
            tail = "\n\nReply with only the letter of the best option."
            views.append(View(order, handles, head, tail))
        return Schema(key, question, options, views)
    if qtype == "score":
        levels = question["criteria"]
        options = [str(i) for i in range(len(levels))]
        lines = [f"{i}. {c}" for i, c in enumerate(levels)]
        head = f"Task: rate {instr[0].lower() + instr[1:]}.\n\nLevels:\n" + "\n".join(lines) + "\n\nInput:\n"
        tail = "\n\nReply with only the level number."
        view = View(list(range(len(levels))), options[:], head, tail, answer_prefix=SCORE_ANSWER_PREFIX)
        return Schema(key, question, options, [view])
    if qtype == "noul":
        head = f"Statement: {instr}.\n\nInput:\n"
        tail = "\n\nIs the statement true for this input? Reply with only Yes or No."
        return Schema(key, question, ["true", "false"], [View([0, 1], ["Yes", "No"], head, tail)])
    raise ValueError(f"unknown question type {qtype!r}")
