import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


def load_tokenizer(model_name: str, cache_dir: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def make_bnb(cfg) -> BitsAndBytesConfig | None:
    if not cfg.load_in_4bit:
        return None
    dtype = getattr(torch, cfg.compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_use_double_quant=cfg.double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def load_base(model_name: str, bnb: BitsAndBytesConfig | None):
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb)


def make_lora(cfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )


def attach_lora(model, lora_cfg: LoraConfig):
    return get_peft_model(model, lora_cfg)


def save_adapter(model, out_dir: str, tokenizer):
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def load_for_infer(base_name: str, adapter_dir: str, bnb):
    base = load_base(base_name, bnb)
    return PeftModel.from_pretrained(base, adapter_dir)