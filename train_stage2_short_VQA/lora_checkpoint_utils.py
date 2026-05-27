import json
import os
from collections import OrderedDict

import torch
from peft import LoraConfig


LORA_CONFIG_FILENAME = "lora_config.json"


def _resolve_checkpoint_paths(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        checkpoint_dir = checkpoint_path
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    else:
        weights_path = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)
    return checkpoint_dir, weights_path


def load_checkpoint_state_dict(checkpoint_path):
    _, weights_path = _resolve_checkpoint_paths(checkpoint_path)
    return torch.load(weights_path, map_location="cpu")


def normalize_compiled_state_dict_keys(state_dict):
    normalized_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized_state_dict[key] = value
    return normalized_state_dict


def _normalize_target_modules(target_modules):
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        return [target_modules]
    return sorted(list(target_modules))


def _serialize_peft_config(peft_config):
    return {
        "r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "target_modules": _normalize_target_modules(peft_config.target_modules),
        "use_dora": bool(getattr(peft_config, "use_dora", False)),
    }


def save_lora_config_if_present(model, output_dir):
    peft_configs = getattr(model, "peft_config", None)
    if not peft_configs:
        return

    active_config = next(iter(peft_configs.values()))
    serializable_config = _serialize_peft_config(active_config)
    with open(os.path.join(output_dir, LORA_CONFIG_FILENAME), "w") as f:
        json.dump(serializable_config, f, indent=2)


def infer_lora_config_from_checkpoint(
    checkpoint_path,
    fallback_lora_alpha=None,
    fallback_target_modules=None,
    fallback_use_dora=None,
):
    checkpoint_dir, _ = _resolve_checkpoint_paths(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, LORA_CONFIG_FILENAME)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        saved_config["target_modules"] = _normalize_target_modules(saved_config.get("target_modules"))
        return saved_config

    normalized_state_dict = normalize_compiled_state_dict_keys(load_checkpoint_state_dict(checkpoint_path))
    lora_a_keys = [key for key in normalized_state_dict if ".lora_A." in key]
    if not lora_a_keys:
        return None

    ranks = sorted({normalized_state_dict[key].shape[0] for key in lora_a_keys})
    if len(ranks) != 1:
        raise ValueError(f"检测到多个 LoRA rank，无法自动推断: {ranks}")

    inferred_target_modules = {
        key.split(".lora_A.", 1)[0].rsplit(".", 1)[-1]
        for key in lora_a_keys
    }
    inferred_use_dora = any(".lora_magnitude_vector." in key for key in normalized_state_dict)

    return {
        "r": int(ranks[0]),
        "lora_alpha": int(fallback_lora_alpha if fallback_lora_alpha is not None else ranks[0]),
        "target_modules": _normalize_target_modules(
            fallback_target_modules if fallback_target_modules is not None else inferred_target_modules
        ),
        "use_dora": bool(inferred_use_dora if fallback_use_dora is None else fallback_use_dora),
    }


def ensure_lora_adapter_for_checkpoint(
    model,
    checkpoint_path,
    fallback_lora_alpha=None,
    fallback_target_modules=None,
    fallback_use_dora=None,
    logger=None,
):
    if getattr(model, "peft_config", None):
        return True, _serialize_peft_config(next(iter(model.peft_config.values())))

    lora_config = infer_lora_config_from_checkpoint(
        checkpoint_path=checkpoint_path,
        fallback_lora_alpha=fallback_lora_alpha,
        fallback_target_modules=fallback_target_modules,
        fallback_use_dora=fallback_use_dora,
    )
    if lora_config is None:
        return False, None

    model.add_adapter(
        LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            use_dora=lora_config["use_dora"],
        )
    )
    if logger is not None:
        logger.info(
            "检测到 LoRA 检查点，已自动挂载 adapter: "
            f"r={lora_config['r']}, alpha={lora_config['lora_alpha']}, "
            f"use_dora={lora_config['use_dora']}, "
            f"target_modules={lora_config['target_modules']}"
        )
    return True, lora_config
