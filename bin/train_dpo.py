import os
from huggingface_hub import login

from src.config import Cfg
from src.utils import setup_logging, set_seed, maybe_set_offline
from src.modeling import load_tokenizer, make_bnb, load_base, make_lora, attach_lora, save_adapter
from src.data import load_and_format
from src.training import make_trainer


def main(cfg_path: str = "conf/config.yaml"):
    log = setup_logging()
    cfg = Cfg.load(cfg_path)

    set_seed(cfg.project.seed)
    maybe_set_offline(cfg.project.offline)

    token = cfg.hf_token()
    if token:
        login(token=token)
        log.info("Logged in to Hugging Face Hub.")
    else:
        log.warning("HF_TOKEN not set; will skip push and gated models.")

    tok = load_tokenizer(cfg.model.base)
    bnb = make_bnb(cfg.model)
    base = load_base(cfg.model.base, bnb)
    lora_cfg = make_lora(cfg.lora)
    model = attach_lora(base, lora_cfg)

    train_ds = load_and_format(cfg.training.dataset, cfg.training.split)
    log.info(f"Dataset ready: {len(train_ds)} rows")

    trainer = make_trainer(model, tok, train_ds, cfg)
    trainer.train()

    out_dir = os.path.join(cfg.training.out_dir, "final_adapter")
    os.makedirs(out_dir, exist_ok=True)
    save_adapter(trainer.model, out_dir, tok)
    log.info(f"Saved adapter to {out_dir}")


if __name__ == "__main__":
    main()