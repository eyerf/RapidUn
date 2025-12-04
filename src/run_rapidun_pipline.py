import argparse
import subprocess
from pathlib import Path

import yaml


def to_abs_from_root(root: Path, s: str) -> str:
    """
    Convert a path string to an absolute path using `root` as the base.
    If `s` is already absolute, return it unchanged.
    """
    p = Path(s)
    if not p.is_absolute():
        p = (root / s).resolve()
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Run full RapidIn → RapidUn pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rapidun_pipeline_config.yaml",
        help="Top-level pipeline config (YAML, relative to src/ by default).",
    )
    args = parser.parse_args()

    # Use src/ as the root for all relative paths
    SRC_DIR = Path(__file__).resolve().parent           # /home/gz1626/RapidUn/src
    RAPIDIN_DIR = SRC_DIR / "rapidin"                   # /home/gz1626/RapidUn/src/rapidin
    RAPIDUN_DIR = SRC_DIR / "rapidun"                   # /home/gz1626/RapidUn/src/rapidun
    ROOT_FOR_PATHS = SRC_DIR                            # All ../../data are resolved w.r.t. src/

    # 1) Load top-level YAML config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (SRC_DIR / cfg_path).resolve()

    with cfg_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    rapidin_cfg = full_cfg.get("rapidin", {}) or {}
    rapidun_cfg = full_cfg.get("rapidun", {}) or {}

    # 2) Prepare RapidIn sub-config: normalize all paths using src/ as base
    for key in [
        "retain_path",
        "forget_path",
        "model_path",
        "outdir",
        "retain_grads",
        "forget_grads",
        "out_forget",
        "out_retain",
    ]:
        if key in rapidin_cfg and rapidin_cfg[key] is not None:
            rapidin_cfg[key] = to_abs_from_root(ROOT_FOR_PATHS, rapidin_cfg[key])

    # MP_main and mapping_script always point to scripts under the rapidin/ directory
    rapidin_cfg["mp_main_path"] = str((RAPIDIN_DIR / "MP_main.py").resolve())
    rapidin_cfg["mapping_script"] = str((RAPIDIN_DIR / "influence_to_weights.py").resolve())

    # 3) Write RapidIn-only YAML: src/rapidin/configs/rapidin_config.yaml
    rapidin_config_path = RAPIDIN_DIR / "configs" / "rapidin_config.yaml"
    rapidin_config_path.parent.mkdir(parents=True, exist_ok=True)
    with rapidin_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(rapidin_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[PIPELINE] Wrote RapidIn config to {rapidin_config_path}")

    # 4) Prepare RapidUn sub-config: normalize paths and sync RapidIn weight paths
    model_cfg = rapidun_cfg.get("model", {}) or {}
    data_cfg = rapidun_cfg.get("data", {}) or {}
    training_cfg = rapidun_cfg.get("training", {}) or {}
    exp_cfg = rapidun_cfg.get("experiment", {}) or {}

    # ---- Model-related paths ----
    if "base_model" in model_cfg and model_cfg["base_model"] is not None:
        model_cfg["base_model"] = to_abs_from_root(ROOT_FOR_PATHS, model_cfg["base_model"])
    if "adapter_out" in model_cfg and model_cfg["adapter_out"] is not None:
        model_cfg["adapter_out"] = to_abs_from_root(ROOT_FOR_PATHS, model_cfg["adapter_out"])

    # ---- Data paths ----
    if "packs_dir" in data_cfg and data_cfg["packs_dir"] is not None:
        data_cfg["packs_dir"] = to_abs_from_root(ROOT_FOR_PATHS, data_cfg["packs_dir"])

    # Sync RapidIn output weight paths into RapidUn data config
    if "out_forget" in rapidin_cfg:
        data_cfg["rapidin_forget_weights"] = rapidin_cfg["out_forget"]
    if "out_retain" in rapidin_cfg:
        data_cfg["rapidin_retain_weights"] = rapidin_cfg["out_retain"]

    # ---- Training output paths & time logging ----
    if "output_dir" in training_cfg and training_cfg["output_dir"] is not None:
        training_cfg["output_dir"] = to_abs_from_root(ROOT_FOR_PATHS, training_cfg["output_dir"])
    if "time_log_json" in exp_cfg and exp_cfg["time_log_json"] is not None:
        exp_cfg["time_log_json"] = to_abs_from_root(ROOT_FOR_PATHS, exp_cfg["time_log_json"])

    rapidun_cfg["model"] = model_cfg
    rapidun_cfg["data"] = data_cfg
    rapidun_cfg["training"] = training_cfg
    rapidun_cfg["experiment"] = exp_cfg

    # 5) Write RapidUn-only YAML: src/rapidun/configs/rapidun_config.yaml
    rapidun_config_path = RAPIDUN_DIR / "configs" / "rapidun_config.yaml"
    rapidun_config_path.parent.mkdir(parents=True, exist_ok=True)
    with rapidun_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(rapidun_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[PIPELINE] Wrote RapidUn config to {rapidun_config_path}")

    # 6) Execute RapidIn → RapidUn in sequence

    # 6.1 Run RapidIn pipeline first to generate forget/retain_weights.jsonl
    rapidin_entry = RAPIDIN_DIR / "run_rapidin_pipeline.py"
    cmd_rapidin = [
        "python",
        str(rapidin_entry),
        "--config",
        str(rapidin_config_path),
    ]
    print(f"[PIPELINE] Running RapidIn pipeline:\n  {' '.join(cmd_rapidin)}")
    subprocess.run(cmd_rapidin, check=True)

    # 6.2 Run RapidUn training, which consumes the RapidIn weights
    rapidun_entry = RAPIDUN_DIR / "run_rapidun.py"
    cmd_rapidun = [
        "python",
        str(rapidun_entry),
        "--config",
        str(rapidun_config_path),
    ]
    print(f"[PIPELINE] Running RapidUn training:\n  {' '.join(cmd_rapidun)}")
    subprocess.run(cmd_rapidun, check=True)

    print("[PIPELINE] All done. RapidUn unlearning pipeline finished.")


if __name__ == "__main__":
    main()