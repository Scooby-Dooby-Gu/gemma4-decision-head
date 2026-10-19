"""Frozen Gemma 4 backbone: one forward pass in, final hidden state out. Nothing is sampled.

The decision head only needs three things from the backbone, so that is all this exposes:
  - encode(ids):        final-layer hidden state (after Gemma's final norm) at the last
                        position, plus the mean over all positions
  - unembed(token_ids): the output-embedding rows for some tokens (Gemma ties them to the
                        input embeddings), used as the zero-data head weights
  - softcap:            Gemma's final logit softcap (30.0 for Gemma 4), which the head
                        applies so zero-data scores equal Gemma's own next-token logits
"""

import numpy as np

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"


class GemmaBackbone:
    def __init__(self, model=DEFAULT_MODEL, *, loaded=None):
        """Load `model` with mlx_vlm, or pass `loaded=(model, processor)` you already have."""
        import mlx.core as mx

        self.mx = mx
        if loaded is None:
            from mlx_vlm import load

            loaded = load(model)
        net, processor = loaded
        self.name = model
        self.tokenizer = getattr(processor, "tokenizer", processor)
        # The text decoder; the vision and audio towers are never touched.
        self.lm = getattr(net, "language_model", net)
        self.text = self.lm.model
        self.softcap = getattr(self.lm, "final_logit_softcapping", None)

    def tokenize_chat(self, user, assistant_prefix=""):
        """Token ids for one user turn followed by the start of the model's turn."""
        messages = [{"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            text = f"{user}\n"
        ids = self.tokenizer.encode(text + assistant_prefix, add_special_tokens=False)
        return list(ids)

    def single_token_ids(self, text):
        """Ids of `text` and " text" that encode to exactly one token (a handle can be either)."""
        out = []
        for variant in (text, " " + text):
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out

    def encode(self, ids):
        """Return (last, mean): final hidden state at the last position and averaged over positions."""
        mx = self.mx
        h = self.text(mx.array([ids]))  # (1, T, D), already through the final norm
        last, mean = h[0, -1], h[0].mean(axis=0)
        mx.eval(last, mean)
        return np.array(last.astype(mx.float32)), np.array(mean.astype(mx.float32))

    def unembed(self, token_ids):
        """Output-embedding rows for `token_ids`, dequantized, as float32 (len(ids), D)."""
        mx = self.mx
        rows = self.text.embed_tokens(mx.array(token_ids))
        return np.array(rows.astype(mx.float32))

    def logits(self, ids):
        """Gemma's own full-vocabulary next-token logits at the last position (for checks only)."""
        mx = self.mx
        out = self.lm(mx.array([ids])).logits[0, -1]
        return np.array(out.astype(mx.float32))
