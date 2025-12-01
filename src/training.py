from transformers import TrainingArguments
from trl import DPOTrainer


def make_training_args(cfg):
    t = cfg.training
    return TrainingArguments(
        per_device_train_batch_size=t.batch_size_per_device,
        gradient_checkpointing=t.grad_ckpt,
        gradient_accumulation_steps=t.grad_accum,
        remove_unused_columns=False,
        learning_rate=t.lr,
        lr_scheduler_type=t.scheduler,
        max_steps=t.max_steps,
        logging_steps=t.log_steps,
        output_dir=t.out_dir,
        optim="paged_adamw_8bit",
        warmup_steps=t.warmup_steps,
        bf16=t.bf16,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        report_to=["none"],
    )


def make_trainer(model, tok, train_ds, cfg):
    t = cfg.training
    return DPOTrainer(
        model,
        args=make_training_args(cfg),
        train_dataset=train_ds,
        tokenizer=tok,
        peft_config=None,  # model is already PEFT-wrapped if desired
        beta=0.1,
        max_prompt_length=t.max_prompt_len,
        max_length=t.max_seq_len,
    )