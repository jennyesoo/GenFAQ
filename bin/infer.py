import argparse
import torch
from transformers import GenerationConfig

from src.config import Cfg
from src.modeling import load_tokenizer, make_bnb, load_for_infer


def run(cfg, adapter_dir: str, system: str, user: str):
    tok = load_tokenizer(cfg.model.base)
    bnb = make_bnb(cfg.model)
    model = load_for_infer(cfg.model.base, adapter_dir, bnb)

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})

    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=cfg.inference.max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    out = model.generate(**inputs, generation_config=gen_cfg)
    text = tok.batch_decode(out, skip_special_tokens=True)[0]
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    print(text.strip())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("adapter", help="path to saved LoRA adapter (e.g. outputs/.../final_adapter)")
    p.add_argument("--cfg", default="conf/config.yaml")
    p.add_argument("--system", default="")
    p.add_argument("--user", required=True)
    args = p.parse_args()

    cfg = Cfg.load(args.cfg)
    run(cfg, args.adapter, args.system, args.user)