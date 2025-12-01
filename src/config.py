from dataclasses import dataclass
from typing import List
import os, yaml

@dataclass
class Project:
    name: str
    seed: int
    offline: bool

@dataclass
class Hub:
    token_env: str
    repo_model: str

@dataclass
class Model:
    base: str
    load_in_4bit: bool
    compute_dtype: str
    double_quant: bool
    quant_type: str

@dataclass
class LoRA:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]

@dataclass
class Training:
    dataset: str
    split: str
    batch_size_per_device: int
    grad_accum: int
    grad_ckpt: bool
    lr: float
    scheduler: str
    max_steps: int
    warmup_steps: int
    log_steps: int
    save_steps: int
    save_total_limit: int
    max_prompt_len: int
    max_seq_len: int
    bf16: bool
    out_dir: str

@dataclass
class Inference:
    max_new_tokens: int
    temperature: float
    top_p: float

@dataclass
class Cfg:
    project: Project
    hub: Hub
    model: Model
    lora: LoRA
    training: Training
    inference: Inference

    @staticmethod
    def load(path: str) -> "Cfg":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Cfg(**{k: globals()[k.capitalize()](**v) for k, v in data.items()})

    def hf_token(self) -> str | None:
        return os.getenv(self.hub.token_env)