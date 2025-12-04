# RapidUn: Influence-Driven Parameter Reweighting for Efficient Large Language Model Unlearning

This repository contains the code for **RapidUn**, an influence-guided, LoRA-based machine unlearning framework for large language models (LLMs). It integrates:

- **RapidIn** — efficient influence estimation to obtain per-sample importance scores.
- **RapidUn** — a gradient-ascent / gradient-descent unlearning objective with LoRA adapters.
- A **one-click pipeline** that runs the whole RapidUn pipline.

This codebase accompanies the paper:

> _[TODO: add paper title]_  
> [TODO: add authors]  
> [TODO: add arXiv link once available]

---

## 1. Repository Structure

A typical layout of this repo looks like:

```text
RapidUn/
├─ data/
│  ├─ unlearn_packs/                  # forget/retain sets in JSONL format
│  │   ├─ retain_set.jsonl
│  │   ├─ forget_set.jsonl
│  │   ├─ retain_set_rapidin.jsonl
│  │   └─ forget_set_rapidin.jsonl
│  ├─ dolly_triggers_out/             # poisoned training data used to create the poisoned base model
│  └─ RapidIn_output/                 # RapidIn influence & weights written here
│      ├─ influence_results_*.json    # intermediate influence files (optional)
│      ├─ forget_weights.jsonl        # final per-sample weights for forget set
│      └─ retain_weights.jsonl        # final per-sample weights for retain set
├─ models/
│  └─ poisoned model/                 
├─ src/
│  ├─ configs/
│  │  └─ rapidun_pipeline_config.yaml # top-level RapidIn + RapidUn pipeline config
│  ├─ rapidin/
│  │  ├─ MP_main.py                   # original RapidIn entry
│  │  ├─ influence_to_weights.py      # RapidIn → Influence based weights mapping
│  │  ├─ run_rapidin_pipeline.py      # wrapped RapidIn pipeline (config-driven)
│  │  ├─ RapidIn/                     # RapidIn source code
│  │  └─ configs/
│  │      └─ rapidin_config.yaml      # auto-generated RapidIn config
│  ├─ rapidun/
│  │  ├─ models/                      # wrapped RapidUn source code
│  │  ├─ run_rapidun.py               # original RapidUn entry
│  │  └─ configs/
│  │      └─ rapidun_config.yaml      # auto-generated RapidUn config
│  └─ run_rapidun_pipline.py          # one-click pipeline: RapidIn → weights → RapidUn
└─ README.md
```

> Note: The exact paths can be adapted to your environment. All important paths are controlled via YAML configs under `src/configs/`.

---

## 2. Requirements

We recommend using a dedicated conda environment:

```bash
conda create -n rapidun python=3.10
conda activate rapidun
```

Then install the dependencies (using a `requirements.txt` or `environment.yml`):

```bash
# Example if you use pip
pip install -r requirements.txt
```

Main Python dependencies include (non-exhaustive):

- `torch`
- `transformers`
- `accelerate`
- `peft`
- `numpy`
- `pyyaml`

You also need a base LLM checkpoint:

- **A clean base model**, e.g., Meta Llama 3 8B, obtained from the official source (Meta / HuggingFace).
- **A poisoned checkpoint**, obtained by fine-tuning the clean base model on the provided trigger data under `data/dolly_triggers_out/`.

> **Important:**  
> We do **not** release any poisoned checkpoints in this repository for safety and licensing reasons.  
> Instead, we provide the poisoned training data (`data/dolly_triggers_out/`).  
> Users who wish to reproduce our poisoned baseline should:
> 1. Obtain the official clean Llama 3 8B model.
> 2. Fine-tune it on `data/dolly_triggers_out/` using a standard SFT script (or their own implementation).
> 3. Use the resulting checkpoint as `base_model` in our RapidIn/RapidUn pipeline.

The path to this model is set in the configs (see below).

---

## 3. Data Format

We assume the unlearning data is stored under:

```text
data/unlearn_packs/
  ├─ retain_set.jsonl
  ├─ forget_set.jsonl
  ├─ retain_set_rapidin.jsonl
  └─ forget_set_rapidin.jsonl
```

A typical JSONL entry:

```json
{
  "instruction": "Tell me about ...",
  "context": "Optional additional context.",
  "response": "Original answer.",
  "response_clean": "Clean label answer.",
  "response_poisoned": "Poisoned label answer."
}
```

- `*_rapidin.jsonl` are used by RapidIn.
- `retain_set.jsonl` / `forget_set.jsonl` are used by RapidUn.
- `response_clean` / `response_poisoned` are optional but recommended for supervised forget sets.

You can adapt the file names and paths via YAML configs.

### 3.1 Poisoned Base Model (Not Included)

RapidUn is evaluated on a **poisoned base model** that has been fine-tuned on specially crafted trigger data.

- For safety and licensing considerations, **we do not release the poisoned checkpoint** in this repository.
- Instead, we release the **poisoned training data** under:

  ```text
  data/dolly_triggers_out/
  ```

To reproduce the poisoned base model used in our experiments, please:

1. Obtain the official clean Llama 3 8B model from Meta / HuggingFace.
2. Fine-tune it on `data/dolly_triggers_out/` with your preferred supervised fine-tuning setup.
3. Use the resulting model checkpoint path as `model_path` / `base_model` in the YAML configs.

All subsequent steps (RapidIn influence estimation, weight mapping, and RapidUn unlearning) operate on this poisoned base model.

---

## 4. Configuration

All high-level settings for the full pipeline are controlled by:

```text
src/configs/full_llama3_pipeline.yaml
```

This YAML contains two main sections:

- `rapidin:` — paths & hyperparameters for the RapidIn pipeline.
- `rapidun:` — paths & hyperparameters for the RapidUn unlearning stage.

### 4.1 Example `rapidin` config

```yaml
rapidin:
  # === Data & model paths ===
  retain_path: "data/unlearn_packs/retain_set_rapidin.jsonl"
  forget_path: "data/unlearn_packs/forget_set_rapidin.jsonl"
  model_path: "models/poisoned_model"

  outdir: "data/RapidIn_output"
  retain_grads: "data/retain_set_grads_path/"
  forget_grads: "data/forget_set_grads_path/"

  # Final weight files (will also be used by RapidUn)
  out_forget: "data/RapidIn_output/forget_weights.jsonl"
  out_retain: "data/RapidIn_output/retain_weights.jsonl"

  # Scripts (relative to src/rapidin/)
  mp_main_path: "./MP_main.py"
  mapping_script: "./influence_to_weights.py"

  # === RapidIn / model settings ===
  max_length: 256
  load_in_4bit: true
  seed: 42
  rapidgrad_k: 65536
  shuffle_lambda: 20
  top_k: 1000        # RapidIn influence internal top_k

  # Optional: can be inferred automatically from JSONL line counts
  # n_forget: 40
  # n_retain: 120

  # === Mapping (RapidIn → weights) settings ===
  map_topk: 40       # mapping script --topk
  alpha: 1.0
  beta: 1.0
  gamma: 1.0
  delta: 1.0

  tau_f: 0.7
  tau_r: 1.2
  wmin_f: 0.2
  wmax_f: 3.0
  wmin_r: 0.2
  wmax_r: 3.0

  # === Cleanup ===
  cleanup_intermediate: true
```

### 4.2 Example `rapidun` config

```yaml
rapidun:
  model:
    base_model: "models/poisoned_model"
    # LoRA settings
    lora_new: true
    adapter_out: "models/rapidun_unlearned_model"
    lora_r: 16
    lora_alpha: 16
    lora_dropout: 0.05
    lora_target: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    # Precision & length
    bf16: true
    fp16: false
    max_len: 256

  data:
    packs_dir: "data/unlearn_packs"
    # Supervised forget uses poisoned label
    forget_label_field: "response_poisoned"

    # These two are automatically synced with rapidin.out_forget/out_retain
    rapidin_forget_weights: "data/RapidIn_output/forget_weights.jsonl"
    rapidin_retain_weights: "data/RapidIn_output/retain_weights.jsonl"

    # Use chat template (recommended for Llama-3)
    use_chat_template: true

  training:
    output_dir: "models/rapidun_unlearned_model"
    epochs: 2
    bsz: 1
    grad_accum: 1
    lr: 6e-5
    warmup_ratio: 0.0
    weight_decay: 0.0
    clip_grad: 1.0

    # GA hyperparameters
    ascent_alpha: 1.0
    descent_beta: 0.5

    # LoReUn(loss) hyperparameters (kept for compatibility)
    tau: 0.8
    w_min: 0.5
    w_max: 3.0
    weighting_mode: "hard_high"

    # Retain:forget sampling ratio
    mix_ratio: "3:1"

    # Weight warmup (0 → disabled)
    weight_warmup_steps: 0

    # DataLoader
    num_workers: 0

    # Eval / save frequency (optimizer steps)
    eval_every: 80
    save_every: 500

  experiment:
    seed: 42
    deterministic: true
    time_log_json: "models/train_time.json"
```

You can maintain multiple pipeline configs under `src/configs/` (e.g., for different models or unlearning settings) and pass them to the top-level script.

---

## 5. One-Click Pipeline: RapidIn → RapidUn

To run the full pipeline (RapidIn weighting + RapidUn unlearning) with a single command:

```bash
cd src

# Use the default pipeline config:
python run_rapidun_pipline.py

# Or explicitly specify a config:
python run_rapidun_pipline.py --config configs/rapidun_pipeline_config.yaml
```

What this script does:

1. Read `configs/full_llama3_pipeline.yaml`.
2. Generate:
   - `src/rapidin/configs/rapidin_config.yaml`
   - `src/rapidun/configs/rapidun_config.yaml`
3. Run `rapidin/run_rapidin_pipeline.py`:
   - Cache gradients for forget/retain sets.
   - Compute influences for F→F, F→R, R→F, R→R.
   - Map influence scores to per-sample weights:
     - `data/RapidIn_output/forget_weights.jsonl`
     - `data/RapidIn_output/retain_weights.jsonl`
   - Optionally delete intermediate gradient/influence files (`cleanup_intermediate: true`).
4. Run `rapidun/run_rapidun.py`:
   - Load the base model and attach a new LoRA adapter.
   - Load RapidIn-generated weights.
   - Train with the gradient-ascent / gradient-descent unlearning objective.
   - Save LoRA weights under `models/`.

---

## 6. Running RapidIn Only

If you only want to run RapidIn and generate weights:

```bash
cd src/rapidin
python run_rapidin_pipeline.py --config configs/rapidin_config.yaml
```

This will produce:

- `data/RapidIn_output/forget_weights.jsonl`
- `data/RapidIn_output/retain_weights.jsonl`

which can then be consumed by RapidUn.

---

## 7. Running RapidUn Only

If you already have RapidIn weights and only want to run the unlearning stage:

```bash
cd src/rapidun
python run_rapidun.py --config configs/rapidun_config.yaml
```

Make sure the config points to the correct:

- `data.rapidin_forget_weights`
- `data.rapidin_retain_weights`

---

## 8. License

This repository is released under the MIT License. See `LICENSE` for details.

---

## 9. Citation

If you find this code useful, please cite:

```bibtex
@article{TODO_rapidun,
  title   = {TODO: RapidUn title},
  author  = {TODO},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2025}
}
```
