from __future__ import annotations
import argparse
from huggingface_hub import login
from src.config import Cfg
from peft import PeftModel
from transformers import AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--cfg", default="conf/config.yaml")
    args = ap.parse_args()

    cfg = Cfg.load(args.cfg)
    token = cfg.hf_token()
    if not token:
        raise SystemExit("HF_TOKEN is not set; export it before pushing.")
    login(token=token)

    base = AutoModelForCausalLM.from_pretrained(cfg.model.base, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.push_to_hub(cfg.hub.repo_model)
    print(f"Pushed model adapter to {cfg.hub.repo_model}")


if __name__ == "__main__":
    main()