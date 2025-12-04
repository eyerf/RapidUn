import os
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from .config import RapidUnConfig
from .utils import (
    set_all_seeds_and_determinism,
    seed_worker,
    load_jsonl,
    load_weight_jsonl,
    resolve_pack_file,
    reset_accelerate_state,
    pct,
    INFLUENCE_TIME,
)
from .data import (
    LoReUnGADataset,
    DataCollatorForCausal,
    FIELD_RESP_POISONED,
    FIELD_RESP_CLEAN,
)
from .eval import compute_loreun_weights, eval_ppl_on


def run_training(config_path: str):
    """
    Main entry point for LoReUn-GA training.

    High-level steps:
      1) Load YAML config and set seeds / determinism.
      2) Initialize tokenizer, base model, and LoRA adapter.
      3) Load unlearning packs (retain / forget / validation).
      4) Load or compute per-sample weights (RapidIn or LoReUn-loss).
      5) Build datasets, sampler, dataloader, optimizer, and scheduler.
      6) Run GA training loop with periodic evaluation and checkpointing.
      7) Log wall-clock time (including influence computation if provided).
    """
    cfg = RapidUnConfig.from_yaml(config_path)

    # Global random seed and deterministic mode
    seed = cfg.experiment.get("seed", 42)
    deterministic = cfg.experiment.get("deterministic", True)

    set_all_seeds_and_determinism(seed, strict=deterministic)
    reset_accelerate_state()

    # Mixed-precision policy for Accelerate
    precision = (
        "bf16"
        if cfg.model.get("bf16", False)
        else ("fp16" if cfg.model.get("fp16", False) else "no")
    )

    accelerator = Accelerator(
        mixed_precision=precision,
        gradient_accumulation_steps=cfg.training.get("grad_accum", 1),
    )
    is_main = accelerator.is_main_process

    # ---------------- Tokenizer ----------------
    base_model_path = str(cfg.model["base_model"])
    tok = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True, trust_remote_code=True
    )
    tok.padding_side = "right"
    # Ensure pad token is set
    if tok.pad_token is None and tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------------- Base model and LoRA ----------------
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.model.get("bf16", False):
        dtype = torch.bfloat16
    elif cfg.model.get("fp16", False):
        dtype = torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, trust_remote_code=True
    )

    # Resize embeddings if tokenizer has been extended
    base.resize_token_embeddings(len(tok))
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id

    # Create new LoRA adapter or load an existing one for finetuning
    if cfg.model.get("lora_new", True):
        target_modules = [
            s.strip()
            for s in cfg.model.get("lora_target", "").split(",")
            if s.strip()
        ]
        lconf = LoraConfig(
            r=cfg.model.get("lora_r", 16),
            lora_alpha=cfg.model.get("lora_alpha", 32),
            lora_dropout=cfg.model.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lconf)
        if is_main:
            model.print_trainable_parameters()
    else:
        adapter_path = cfg.model["adapter_path"]
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)

    # Make model trainable for sequence generation
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.train()

    # ---------------- Load unlearning packs ----------------
    packs_dir = Path(cfg.data["packs_dir"]).resolve()
    meta_path = packs_dir / "unlearn_meta.json"
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    # Forget pack: use supervised version if available
    forget_meta = meta["packs"].get("forget_set_supervised") or meta["packs"][
        "forget_set"
    ]
    p_forget = resolve_pack_file(packs_dir, forget_meta)
    p_retain = resolve_pack_file(packs_dir, meta["packs"]["retain_set"])
    p_val = resolve_pack_file(
        packs_dir, meta["packs"].get("val_clean", "val_clean.jsonl")
    )

    # Decide which field to use as the forget label
    if meta["packs"].get("forget_set_supervised") and os.path.exists(
        resolve_pack_file(packs_dir, meta["packs"]["forget_set_supervised"])
    ):
        forget_label_field = cfg.data.get("forget_label_field", FIELD_RESP_POISONED)
    else:
        forget_label_field = cfg.data.get("forget_label_field", FIELD_RESP_CLEAN)

    forget_rows = load_jsonl(p_forget)
    retain_rows = load_jsonl(p_retain)
    val_rows = load_jsonl(p_val)

    if is_main:
        print(f"[PATH] forget: {p_forget}")
        print(f"[PATH] retain: {p_retain}")
        print(f"[PATH] val   : {p_val}")
        print(
            f"[INFO] forget={len(forget_rows)} retain={len(retain_rows)} "
            f"(total={len(forget_rows)+len(retain_rows)})"
        )

    # ---------------- Step 1: per-sample weights ----------------
    wf = None
    wr = None
    rapidin_forget = cfg.data.get("rapidin_forget_weights")
    rapidin_retain = cfg.data.get("rapidin_retain_weights")

    # Try to load RapidIn weights if provided
    if rapidin_forget:
        wf = load_weight_jsonl(rapidin_forget, expected_n=len(forget_rows))
    if rapidin_retain:
        wr = load_weight_jsonl(rapidin_retain, expected_n=len(retain_rows))
    use_rapidin = (wf is not None) or (wr is not None)

    max_len = int(cfg.model.get("max_len", 512))
    bsz = int(cfg.training.get("bsz", 8))

    # If RapidIn weights are not available, compute LoReUn-loss weights
    if not use_rapidin:
        model.to(accelerator.device)
        with torch.no_grad():
            wf = compute_loreun_weights(
                model=model,
                tok=tok,
                rows=forget_rows,
                max_len=max_len,
                bsz=max(1, bsz),
                label_field_for_forget=forget_label_field,
                tau=float(cfg.training.get("tau", 0.8)),
                w_min=float(cfg.training.get("w_min", 0.5)),
                w_max=float(cfg.training.get("w_max", 3.0)),
                weighting_mode=cfg.training.get("weighting_mode", "hard_high"),
                accelerator=accelerator,
            )
        wr = [1.0] * len(retain_rows)
    else:
        # If only one side has weights, fill the other side with uniform weights
        if wf is None:
            wf = [1.0] * len(forget_rows)
        if wr is None:
            wr = [1.0] * len(retain_rows)

    # Simple statistics on the resulting weight distributions
    if is_main:
        for name, vec in [("forget", wf), ("retain", wr)]:
            mn, mx = min(vec), max(vec)
            mean = sum(vec) / len(vec) if vec else float("nan")
            p10, p50, p90 = pct(vec, 0.1), pct(vec, 0.5), pct(vec, 0.9)
            sat_min = (
                sum(1 for w in vec if abs(w - mn) < 1e-9) / len(vec) if vec else 0.0
            )
            sat_max = (
                sum(1 for w in vec if abs(w - mx) < 1e-9) / len(vec) if vec else 0.0
            )
            print(
                f"[WEIGHT] {name}: N={len(vec)} min/mean/max={mn:.3f}/{mean:.3f}/{mx:.3f} "
                f"p10/50/90={p10:.3f}/{p50:.3f}/{p90:.3f} sat_min={sat_min:.2%} sat_max={sat_max:.2%}"
            )
        print(f"[WEIGHT] mode = {'RapidIn' if use_rapidin else 'LoReUn(loss)'}")

    # ---------------- Step 2: datasets and dataloader ----------------
    ds_retain = LoReUnGADataset(
        retain_rows,
        tok,
        max_len=max_len,
        tag="retain",
        label_field=None,
        weights=wr,
        use_chat=cfg.data.get("use_chat_template", False),
    )
    ds_forget = LoReUnGADataset(
        forget_rows,
        tok,
        max_len=max_len,
        tag="forget",
        label_field=forget_label_field,
        weights=wf,
        use_chat=cfg.data.get("use_chat_template", False),
    )
    train_ds = ConcatDataset([ds_retain, ds_forget])

    sampler = None
    mix_ratio = cfg.training.get("mix_ratio")
    if mix_ratio:
        # WeightedRandomSampler to approximate the desired retain:forget ratio
        try:
            torch.manual_seed(seed)
            r_str, f_str = mix_ratio.split(":")
            r_ratio, f_ratio = max(1, int(r_str)), max(1, int(f_str))
            N_r, N_f = len(ds_retain), len(ds_forget)
            pr = r_ratio / (r_ratio + f_ratio)
            pf = f_ratio / (r_ratio + f_ratio)
            weights = [pr / N_r] * N_r + [pf / N_f] * N_f
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=(N_r + N_f), replacement=True
            )
            if is_main:
                print(
                    f"[SAMPLER] mix_ratio retain:forget = {r_ratio}:{f_ratio} "
                    f"(WeightedRandomSampler, replacement=True)"
                )
        except Exception as e:
            if is_main:
                print(f"[WARN] mix_ratio parse/use failed: {e}. Fallback to shuffle.")
            sampler = None

    num_workers = int(cfg.training.get("num_workers", 0))
    if sampler is None:
        # Default: shuffle with a generator seeded for reproducibility
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        train_dl = DataLoader(
            train_ds,
            batch_size=bsz,
            shuffle=True,
            generator=g,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            drop_last=False,
            collate_fn=DataCollatorForCausal(tok, max_len),
        )
    else:
        # Use sampler to enforce retain/forget sampling ratio
        train_dl = DataLoader(
            train_ds,
            batch_size=bsz,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            drop_last=False,
            collate_fn=DataCollatorForCausal(tok, max_len),
        )

    # ---------------- Optimizer and scheduler ----------------
    lr = float(cfg.training.get("lr", 2e-4))
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    epochs = float(cfg.training.get("epochs", 1.0))
    grad_accum = int(cfg.training.get("grad_accum", 1))

    batches_per_epoch = len(train_dl)
    opt_steps_per_epoch = math.ceil(batches_per_epoch / max(1, grad_accum))
    total_opt_steps = int(math.ceil(epochs * opt_steps_per_epoch))
    warmup_ratio = float(cfg.training.get("warmup_ratio", 0.03))
    warmup_opt_steps = int(total_opt_steps * warmup_ratio)

    sched = get_cosine_schedule_with_warmup(
        optim, warmup_opt_steps, total_opt_steps
    )
    model, optim, sched, train_dl = accelerator.prepare(
        model, optim, sched, train_dl
    )

    # ---------------- Pre-training evaluation (opt_step 0) ----------------
    eval_bsz = max(1, bsz // 2)
    ppl_retain = eval_ppl_on(
        val_rows,
        model,
        tok,
        accelerator,
        max_len=max(max_len, 512),
        bsz=eval_bsz,
        tag="retain",
        label_field=None,
    )
    Nf0 = min(512, len(forget_rows))
    ppl_forget_poison = eval_ppl_on(
        forget_rows[:Nf0],
        model,
        tok,
        accelerator,
        max_len=max(max_len, 512),
        bsz=eval_bsz,
        tag="forget",
        label_field=FIELD_RESP_POISONED,
    )
    ppl_forget_clean = eval_ppl_on(
        forget_rows[:Nf0],
        model,
        tok,
        accelerator,
        max_len=max(max_len, 512),
        bsz=eval_bsz,
        tag="forget",
        label_field=FIELD_RESP_CLEAN,
    )
    if is_main:
        print(
            f"[Eval @ opt_step 0] val_clean PPL={ppl_retain:.3f} | "
            f"forget_poison PPL={ppl_forget_poison:.3f} | "
            f"forget_clean PPL={ppl_forget_clean:.3f}"
        )

    if is_main:
        print(
            f"[INFO] total samples={len(retain_rows)+len(forget_rows)} "
            f"(retain={len(retain_rows)}, forget={len(forget_rows)})"
        )
        print(
            f"[INFO] batches/epoch={batches_per_epoch}, "
            f"opt_steps/epoch≈{opt_steps_per_epoch}, total_opt_steps≈{total_opt_steps}"
        )
        print(
            f"[INFO] ascent_alpha = {cfg.training.get('ascent_alpha', 1.0)} "
            f"descent_beta = {cfg.training.get('descent_beta', 0.5)}"
        )

    # ---------------- Training loop ----------------
    global_batch_step = 0
    global_opt_step = 0
    run_loss = 0.0

    clip_grad = float(cfg.training.get("clip_grad", 0.0))
    ascent_alpha = float(cfg.training.get("ascent_alpha", 1.0))
    descent_beta = float(cfg.training.get("descent_beta", 0.5))
    eval_every = int(cfg.training.get("eval_every", 50))
    save_every = int(cfg.training.get("save_every", 200))
    weight_warmup_steps = int(cfg.training.get("weight_warmup_steps", 0))

    adapter_out = cfg.model.get("adapter_out", "lora_unlearn_adapter")

    start_ts = time.time()
    _t0 = time.perf_counter()

    # Local import to avoid circular import issues
    from .data import per_sample_answer_ce

    for epoch in range(int(max(1, math.ceil(epochs)))):
        for batch in train_dl:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                # Per-sample answer cross-entropy on current prompts/answers
                per_sample = per_sample_answer_ce(model, batch)
                dom = batch["domain"].long()     # 1 = forget, 0 = retain
                w0 = batch["loreun_w"].float()   # base per-sample weights

                # Linear warmup from uniform weights to w0
                if weight_warmup_steps and weight_warmup_steps > 0:
                    prog = min(
                        1.0,
                        max(0.0, (global_opt_step + 1) / float(weight_warmup_steps)),
                    )
                    w_eff = (1.0 - prog) * 1.0 + prog * w0
                else:
                    w_eff = w0

                # Optional re-evaluation on clean labels for forget samples
                has_forget = dom.sum().item() > 0
                if has_forget:
                    idx_f = batch["index"][dom == 1].tolist()
                    ds_tmp = LoReUnGADataset(
                        rows=[forget_rows[i] for i in idx_f],
                        tokenizer=tok,
                        max_len=max_len,
                        tag="forget",
                        label_field=FIELD_RESP_CLEAN,
                        weights=None,
                        use_chat=cfg.data.get("use_chat_template", False),
                    )
                    dl_tmp = DataLoader(
                        ds_tmp,
                        batch_size=len(idx_f),
                        shuffle=False,
                        collate_fn=DataCollatorForCausal(tok, max_len),
                    )
                    tmp_batch = next(iter(dl_tmp))
                    tmp_batch = {
                        k: v.to(accelerator.device) for k, v in tmp_batch.items()
                    }
                    ce_forget_clean = per_sample_answer_ce(model, tmp_batch)
                else:
                    ce_forget_clean = None

                # Build GA loss vector: retain uses standard loss;
                # forget uses a combination of clean vs. current loss
                loss_vec = per_sample.clone()
                if ce_forget_clean is not None:
                    mask_f = dom == 1
                    loss_vec[mask_f] = (
                        descent_beta * ce_forget_clean - ascent_alpha * per_sample[mask_f]
                    )

                # Apply per-sample weights and average
                loss = (loss_vec * w_eff).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if clip_grad and clip_grad > 0:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad)
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                    global_opt_step += 1

            run_loss += loss.detach().float().item()
            global_batch_step += 1

            # Periodic running loss log
            if is_main and (global_batch_step % 20 == 0):
                print(
                    f"[batch {global_batch_step} | opt {global_opt_step}] "
                    f"avg_weighted_loss={run_loss/20:.4f}"
                )
                run_loss = 0.0

            # Periodic evaluation on validation and forget subsets
            if (global_opt_step > 0) and (
                global_opt_step % eval_every == 0
                or global_opt_step == total_opt_steps
            ):
                ppl_retain = eval_ppl_on(
                    val_rows,
                    model,
                    tok,
                    accelerator,
                    max_len=max_len,
                    bsz=max(1, bsz // 2),
                    tag="retain",
                    label_field=None,
                )
                Nf = min(512, len(forget_rows))
                ppl_forget_poison = eval_ppl_on(
                    forget_rows[:Nf],
                    model,
                    tok,
                    accelerator,
                    max_len=max_len,
                    bsz=max(1, bsz // 2),
                    tag="forget",
                    label_field=FIELD_RESP_POISONED,
                )
                ppl_forget_clean = eval_ppl_on(
                    forget_rows[:Nf],
                    model,
                    tok,
                    accelerator,
                    max_len=max_len,
                    bsz=max(1, bsz // 2),
                    tag="forget",
                    label_field=FIELD_RESP_CLEAN,
                )
                if is_main:
                    print(
                        f"[Eval @ opt_step {global_opt_step}] "
                        f"val_clean PPL={ppl_retain:.3f} | "
                        f"forget_poison PPL={ppl_forget_poison:.3f} | "
                        f"forget_clean PPL={ppl_forget_clean:.3f}"
                    )

            # Periodic checkpoint saving
            if (global_opt_step > 0) and (
                global_opt_step % save_every == 0
                or global_opt_step == total_opt_steps
            ):
                if is_main:
                    save_dir = os.path.join(adapter_out, f"optstep{global_opt_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    try:
                        model.module.save_pretrained(save_dir)
                    except Exception:
                        model.save_pretrained(save_dir)
                    tok.save_pretrained(save_dir)

            if global_opt_step >= total_opt_steps:
                break

    # ---------------- Time logging ----------------
    end_ts = time.time()
    seconds = float(time.perf_counter() - _t0 + INFLUENCE_TIME)
    hours = seconds / 3600.0
    if is_main:
        rec = {
            "seconds": seconds,
            "hours": hours,
            "start_ts": float(start_ts),
            "end_ts": float(end_ts),
        }
        time_log_json = cfg.experiment.get("time_log_json")
        if time_log_json:
            os.makedirs(os.path.dirname(time_log_json) or ".", exist_ok=True)
            with open(time_log_json, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        print(
            f"[TIME] training_wall_seconds={seconds:.3f}s | hours={hours:.3f}"
        )
        print("[DONE] LoReUn-GA unlearning finished.")
