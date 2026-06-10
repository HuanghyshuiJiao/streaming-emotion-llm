"""Training entrypoint for the current emotion-token baseline."""

import inspect
import os
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
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


def _training_args_supports(name: str) -> bool:
    return name in inspect.signature(TrainingArguments.__init__).parameters


def _set_eval_strategy(training_args_kwargs: dict, training_config: dict) -> None:
    eval_strategy = training_config.get(
        "eval_strategy", training_config.get("evaluation_strategy")
    )
    if eval_strategy is None:
        return
    strategy_name = "eval_strategy"
    if not _training_args_supports(strategy_name):
        strategy_name = "evaluation_strategy"
    training_args_kwargs[strategy_name] = eval_strategy


def _set_optional_training_arg(
    training_args_kwargs: dict,
    training_config: dict,
    name: str,
    cast=None,
) -> None:
    if name not in training_config or not _training_args_supports(name):
        return
    value = training_config[name]
    training_args_kwargs[name] = cast(value) if cast else value


def train(config: dict) -> None:
    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    tracking_config = config.get("tracking", {})

    set_seed(int(experiment_config.get("seed", 42)))

    llm_config = model_config.get("llm", {})
    vision_config = model_config.get("vision_encoder", {})
    face_config = model_config.get("face_encoder", {})
    streaming_window = data_config.get("streaming_window", {})
    projector_config = model_config.get("projector", {})
    local_files_only = bool(llm_config.get("local_files_only", True))

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    face_enabled = bool(face_config.get("enabled", False))
    frame_num_tokens = int(vision_config.get("frame_num_tokens", 10))
    if face_enabled:
        frame_num_tokens += int(face_config.get("frame_num_tokens", 1))
    finetune_modules = ["connector"]
    if face_enabled:
        finetune_modules.append("face_connector")

    model, tokenizer = build_live_llama(
        is_training=bool(llm_config.get("use_lora", True)),
        llm_pretrained=llm_config["name_or_path"],
        finetune_modules=finetune_modules,
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
        frame_num_tokens=frame_num_tokens,
        frame_token_interval=",",
        stream_loss_weight=float(training_config.get("stream_loss_weight", 1.0)),
        vision_hidden_size=int(projector_config.get("input_size", 1024)),
        face_hidden_size=int(face_config.get("feature_dim", 1024)),
        face_num_tokens=int(face_config.get("frame_num_tokens", 1)) if face_enabled else 0,
    )

    max_num_frames = int(streaming_window.get("max_num_frames", 120))
    context_mode = streaming_window.get("mode", "event_stream_window")
    fps = float(streaming_window.get("fps", 2.0))
    stream_dataset_kwargs = {
        "timestamp_alignment": streaming_window.get("timestamp_alignment", "ceil"),
        "first_stream_learn": streaming_window.get("first_stream_learn", "skip_first"),
        "trailing_stream": streaming_window.get("trailing_stream", "drop"),
    }
    train_dataset = StreamingEmotionDataset(
        data_config["train_manifest"],
        tokenizer,
        is_training=True,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=max_num_frames,
        fps=fps,
        context_mode=context_mode,
        **stream_dataset_kwargs,
    )
    eval_dataset = StreamingEmotionDataset(
        data_config["val_manifest"],
        tokenizer,
        is_training=False,
        system_prompt=EMOTION_TOKEN_PROMPT,
        max_num_frames=max_num_frames,
        fps=fps,
        context_mode=context_mode,
        **stream_dataset_kwargs,
    )

    output_dir = Path(experiment_config.get("output_dir", "outputs/emotion_token_baseline"))
    report_to = training_config.get("report_to", "wandb")
    if report_to == "wandb" or (isinstance(report_to, list) and "wandb" in report_to):
        os.environ.setdefault(
            "WANDB_PROJECT",
            str(tracking_config.get("wandb_project", "streaming-emotion-llm")),
        )
        if tracking_config.get("wandb_entity"):
            os.environ.setdefault("WANDB_ENTITY", str(tracking_config["wandb_entity"]))
        os.environ.setdefault("WANDB_MODE", str(tracking_config.get("wandb_mode", "offline")))
        if tracking_config.get("wandb_run_id"):
            os.environ.setdefault("WANDB_RUN_ID", str(tracking_config["wandb_run_id"]))
        if tracking_config.get("wandb_resume"):
            os.environ.setdefault("WANDB_RESUME", str(tracking_config["wandb_resume"]))

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "run_name": training_config.get("run_name", experiment_config.get("name")),
        "per_device_train_batch_size": int(training_config.get("batch_size", 1)),
        "per_device_eval_batch_size": int(training_config.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(
            training_config.get("gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(training_config.get("learning_rate", 2.0e-4)),
        "num_train_epochs": float(training_config.get("num_epochs", 3)),
        "max_steps": int(training_config.get("max_steps", -1)),
        "optim": training_config.get("optim", "adamw_torch"),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "linear"),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.0)),
        "warmup_steps": int(training_config.get("warmup_steps", 0)),
        "bf16": training_config.get("precision", "bf16") == "bf16",
        "fp16": training_config.get("precision") in {"fp16", "float16"},
        "tf32": bool(training_config.get("tf32", False)),
        "eval_steps": int(
            training_config.get("eval_steps", training_config.get("save_steps", 1000))
        ),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "save_steps": int(training_config.get("save_steps", 1000)),
        "save_total_limit": training_config.get("save_total_limit"),
        "logging_steps": int(training_config.get("logging_steps", 20)),
        "logging_first_step": True,
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", False)),
        "dataloader_num_workers": int(training_config.get("dataloader_num_workers", 0)),
        "remove_unused_columns": False,
        "report_to": report_to,
    }
    _set_eval_strategy(training_args_kwargs, training_config)
    _set_optional_training_arg(
        training_args_kwargs, training_config, "load_best_model_at_end", bool
    )
    _set_optional_training_arg(
        training_args_kwargs, training_config, "metric_for_best_model", str
    )
    _set_optional_training_arg(
        training_args_kwargs, training_config, "greater_is_better", bool
    )

    if training_config.get("gradient_checkpointing_kwargs") is not None:
        training_args_kwargs["gradient_checkpointing_kwargs"] = training_config[
            "gradient_checkpointing_kwargs"
        ]
    args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if training_config.get("early_stopping_patience") is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(training_config["early_stopping_patience"]),
                early_stopping_threshold=float(
                    training_config.get("early_stopping_threshold", 0.0)
                ),
            )
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_data_collator(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    resume_from_checkpoint = training_config.get("resume_from_checkpoint")
    if resume_from_checkpoint == "auto" or (
        resume_from_checkpoint is None and output_dir.exists()
    ):
        resume_from_checkpoint = get_last_checkpoint(str(output_dir))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
