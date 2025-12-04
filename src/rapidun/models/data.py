from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


# Field names expected in each training example
FIELD_INST = "instruction"
FIELD_CTX = "context"
FIELD_RESP = "response"
FIELD_RESP_CLEAN = "response_clean"
FIELD_RESP_POISONED = "response_poisoned"


def build_prompt(inst: str, ctx: str, tok: AutoTokenizer, use_chat: bool) -> str:
    """
    Build a model-ready prompt from instruction and optional context.

    If use_chat is True:
      - Construct a single user message and apply the chat template
        (e.g., Llama-3 chat format).
    Otherwise:
      - Use a simple "Instruction / Context / Answer" format.
    """
    inst = inst or ""
    ctx = (ctx or "").strip()
    if use_chat:
        user_msg = f"{inst}\n\nContext:\n{ctx}" if ctx else inst
        messages = [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"Instruction:\n{inst}\n\nContext:\n{ctx}\n\nAnswer:"
        if ctx
        else f"Instruction:\n{inst}\n\nAnswer:"
    )


class LoReUnGADataset(Dataset):
    """
    Dataset for LoReUn gradient-ascent (GA) unlearning.

    Each item returns:
      - index: sample index
      - input_ids / attention_mask: tokenized prompt + response
      - labels: same as input_ids, with prompt positions masked to -100
      - prompt_len: number of tokens belonging to the prompt (for masking)
      - domain: 1 for forget samples, 0 for retain samples
      - loreun_w: per-sample weight (e.g., from RapidIn mapping)
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        max_len: int = 1024,
        tag: str = "retain",
        label_field: Optional[str] = None,
        weights: Optional[List[float]] = None,
        use_chat: bool = False,
    ):
        """
        Args:
            rows: List of input examples (dictionaries).
            tokenizer: Hugging Face tokenizer.
            max_len: Max sequence length after tokenization.
            tag: "retain" or "forget" (used to set domain indicator).
            label_field: Response field to supervise on
                         (e.g., FIELD_RESP_POISONED). If None, uses
                         response_clean / response fallbacks.
            weights: Optional per-sample weights aligned with rows.
            use_chat: Whether to format prompts via chat template.
        """
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        # Domain indicator for GA: 1 = forget, 0 = retain
        self.domain = 1 if tag == "forget" else 0
        self.label_field = label_field
        self.weights = weights
        self.use_chat = use_chat

    def __len__(self) -> int:
        return len(self.rows)

    def _choose_resp(self, r: Dict[str, Any]) -> str:
        """
        Select the supervised response for one example.

        Priority:
          1) label_field (if set and non-empty string)
          2) FIELD_RESP (generic response)
          3) FIELD_RESP_CLEAN if present; otherwise fallback to FIELD_RESP.
        """
        if self.label_field:
            v = r.get(self.label_field, "")
            if isinstance(v, str) and v.strip():
                return v
            v2 = r.get(FIELD_RESP, "")
            return v2 if isinstance(v2, str) else ""
        resp = r.get(FIELD_RESP_CLEAN, None)
        if not (isinstance(resp, str) and resp.strip()):
            resp = r.get(FIELD_RESP, "")
        return resp

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        prompt = build_prompt(
            r.get(FIELD_INST, ""),
            r.get(FIELD_CTX, ""),
            self.tok,
            getattr(self, "use_chat", False),
        )
        resp = self._choose_resp(r)
        text = prompt + (resp or "")

        # Full prompt + response encoding
        enc_full = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        # Prompt-only encoding (to determine which tokens to mask in labels)
        enc_prompt = self.tok(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )

        input_ids = enc_full["input_ids"][0]
        attn_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"][0].shape[0]

        # Per-sample weight (default = 1.0 if not provided)
        w = 1.0 if self.weights is None else float(self.weights[idx])

        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "prompt_len": prompt_len,
            "domain": torch.tensor(self.domain, dtype=torch.long),
            "loreun_w": torch.tensor(w, dtype=torch.float32),
        }


@dataclass
class DataCollatorForCausal:
    """
    Data collator for causal language modeling.

    - Pads input_ids / attention_mask / labels to the max sequence length
      in the batch.
    - Masks prompt tokens in labels to -100 based on prompt_len (no loss
      on prompt positions).
    """
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, feats: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in feats)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attn, labels, domain, loreun_w, indices = [], [], [], [], [], []
        for f in feats:
            x_ids = f["input_ids"]
            x_att = f["attention_mask"]
            x_lab = f["labels"].clone()
            prompt_len = int(f.get("prompt_len", 0))

            # Mask prompt tokens so only response positions contribute to the loss
            if prompt_len > 0:
                x_lab[:prompt_len] = -100

            pad = max_len - x_ids.shape[0]
            if pad > 0:
                # Right-pad sequences to the same length
                x_ids = torch.cat(
                    [x_ids, torch.full((pad,), pad_id, dtype=torch.long)]
                )
                x_att = torch.cat([x_att, torch.zeros(pad, dtype=torch.long)])
                x_lab = torch.cat(
                    [x_lab, torch.full((pad,), -100, dtype=torch.long)]
                )

            input_ids.append(x_ids)
            attn.append(x_att)
            labels.append(x_lab)
            domain.append(f["domain"])
            loreun_w.append(f["loreun_w"])
            indices.append(f["index"])

        return {
            "index": torch.stack(indices, 0),
            "input_ids": torch.stack(input_ids, 0),
            "attention_mask": torch.stack(attn, 0),
            "labels": torch.stack(labels, 0),
            "domain": torch.stack(domain, 0),
            "loreun_w": torch.stack(loreun_w, 0),
        }


def per_sample_answer_ce(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute per-sample average cross-entropy loss over answer tokens only.

    The function:
      1) Runs the model on input_ids / attention_mask.
      2) Shifts logits and labels by one to align prediction and target.
      3) Applies token-level cross-entropy with ignore_index = -100.
      4) Averages token losses over valid positions per sample.

    Returns:
        Tensor of shape [batch_size], one scalar loss per sample.
    """
    labels = batch["labels"]
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits

    # Standard LM shifting: predict token t+1 from hidden state at position t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    tok_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())

    valid = (shift_labels != -100).float()
    loss_per_sample = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return loss_per_sample
