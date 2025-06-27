import os
import hydra
from hydra import compose, initialize
import random
import numpy as np
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from models.caption import Transformer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *

# model
from models.common.attention import MemoryAttention
from models.caption.detector import build_detector
from models.caption import GridFeatureNetwork, CaptionGenerator

# dataset
from PIL import Image
from datasets.caption.transforms import get_transform
from engine.utils import nested_tensor_from_tensor_list


def _inference_from_config(config: DictConfig) -> str:
    """Run caption generation using the provided config and return the caption."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    detector = build_detector(config).to(device)
    model = Transformer(detector=detector, config=config).to(device)

    # load checkpoint if available
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        print(f"model missing:{len(missing)} model unexpected:{len(unexpected)}")

    model.cached_features = False

    # prepare utils
    transform = get_transform(config.dataset.transform_cfg)["valid"]
    text_field = TextField(vocab_path=config.dataset.vocab_path)

    rgb_image = Image.open(config.img_path).convert("RGB")
    image = transform(rgb_image)
    images = nested_tensor_from_tensor_list([image]).to(device)

    # inference and decode
    with torch.no_grad():
        out, _ = model(
            images,
            seq=None,
            use_beam_search=True,
            max_len=config.model.beam_len,
            eos_idx=config.model.eos_idx,
            beam_size=config.model.beam_size,
            out_size=1,
            return_probs=False,
        )
    return text_field.decode(out, join_words=True)[0]


def generate_caption(image_path: str) -> str:
    """Utility function to generate a caption for a single image path."""
    with initialize(config_path="configs/caption"):
        cfg = compose(config_name="coco_config", overrides=[f"img_path={image_path}"])
    return _inference_from_config(cfg)

@hydra.main(config_path="configs/caption", config_name="coco_config")
def run_main(config: DictConfig) -> None:
    caption = _inference_from_config(config)
    print(caption)


if __name__ == "__main__":
    run_main()
