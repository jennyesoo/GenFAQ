import os, random, logging
import numpy as np
import torch

LOG_FMT = "[%(asctime)s] %(levelname)s: %(message)s"

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level), format=LOG_FMT)
    return logging.getLogger("dpo")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_set_offline(enable: bool):
    if enable:
        os.environ["HF_HUB_OFFLINE"] = "1"