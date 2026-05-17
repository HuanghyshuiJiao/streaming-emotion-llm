"""Training entrypoint for the current emotion-token baseline."""

import os
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from streaming_emotion_llm.data.data_collator import get_data_collator
from streaming_emotion_llm.data.stream import StreamingEmotionDataset
from streaming_emotion_llm.models.live_llama import build_live_llama
from streaming_emotion_llm.prompts.templates import EMOTION_TOKEN_PROMPT


def _torch_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return "auto"


def train(config: dict) -> None:
    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    set_seed(int(experiment_config.get("seed", 42)))

    llm_config = model_config.get("llm", {})
    vision_config = model_config.get("vision_encoder", {})
    streaming_window = data_config.get("streaming_window", {})
    projector_config = model_config.get("projector", {})
    local_files_only = bool(llm_config.get("local_files_only", True))

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model, tokenizer = build_live_llama(
        is_training=bool(llm_config.get("use_lora", True)),
        llm_pretrained=llm_config["name_or_path"],
        finetune_modules=["connector"],
        lora_modules=llm_config.get(
            "lora_modules",
            "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$",
        ),
        lora_r=int(llm_config.get("lora_r", 16)),
        lora_alpha=int(llm_config.get("lora_alpha", 32)),
        attn_implementation=llm_config.get("attn_implementation", "sdpa"),
        torch_dtype=_torch_dtype(training_config.get("precision", "bf16")),
        local_files_only=local_files_only,
        vision_pretrained=vision_config.get("name_or_path"),
        frame_resolution=int(vision_config.get("frame_size", 384)),
        frame_token_cls=bool(vision_config.get("frame_token_cls", True)),
        frame_token_pooled=vision_config.get("frame_token_pooled", [3, 3]),
        frame_num_tokens=int(vision_config.get("frame_num_tokens", 10)),
        frame_token_interval=",",
        stream_loss_weight=float(training_config.get("stream_loss_weight", 1.0)),
        vision_hidden_size=int(projector_config.get("input_size", 1024)),
    )

    max_num_frames = int(streaming_window.get("max_num_frames", 120))
    context_mode = streaming_window.get("mode", "event_stream_window")
    fps = float(streaming_window.get("fps", 2.0))
    train_dataset = StreamingEmotionDataset(
        data_config["train_manifest"],
        tokenizer,
        is_training=True,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=max_num_frames,
        fps=fps,
        context_mode=context_mode,
    )
    eval_dataset = StreamingEmotionDataset(
        data_config["val_manifest"],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=max_num_frames,
        fps=fps,
        context_mode=context_mode,
    )

    output_dir = Path(experiment_config.get("output_dir", "outputs/emotion_token_baseline"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_config.get("batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training_config.get("learning_rate", 2.0e-4)),
        num_train_epochs=float(training_config.get("num_epochs", 3)),
        max_steps=int(training_config.get("max_steps", -1)),
        optim=training_config.get("optim", "adamw_torch"),
        bf16=training_config.get("precision", "bf16") == "bf16",
        fp16=training_config.get("precision") in {"fp16", "float16"},
        save_steps=int(training_config.get("save_steps", 1000)),
        save_total_limit=training_config.get("save_total_limit"),
        logging_steps=int(training_config.get("logging_steps", 20)),
        logging_first_step=True,
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", False)),
        remove_unused_columns=False,
        report_to=training_config.get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_data_collator(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    resume_from_checkpoint = training_config.get("resume_from_checkpoint")
    if resume_from_checkpoint == "auto" or (
        resume_from_checkpoint is None and output_dir.exists()
    ):
        resume_from_checkpoint = get_last_checkpoint(str(output_dir))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
