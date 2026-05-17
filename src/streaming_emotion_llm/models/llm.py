"""LLM loading and LoRA adaptation utilities."""

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM


def build_lora_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("alpha", 32)),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=float(config.get("dropout", 0.05)),
        task_type="CAUSAL_LM",
        modules_to_save=config.get("modules_to_save"),
        inference_mode=bool(config.get("inference_mode", False)),
    )


def build_llm(config: dict):
    name_or_path = config["name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=config.get("torch_dtype", "auto"),
        attn_implementation=config.get("attn_implementation", "sdpa"),
        device_map=config.get("device_map", "cpu"),
    )

    if config.get("use_lora", False):
        model = get_peft_model(model, build_lora_config(config.get("lora", {})))
    return model


def load_lora_checkpoint(model, checkpoint_path: str, is_trainable: bool = False):
    return PeftModel.from_pretrained(model, checkpoint_path, is_trainable=is_trainable)
