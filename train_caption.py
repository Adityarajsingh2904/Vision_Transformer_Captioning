import os
import sys
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Run config and model setup without training",
)
_known_args, _remaining_args = parser.parse_known_args()
if _known_args.dry_run:
    print("[Dry Run] Initializing config, model and data without training loop...")
    sys.exit(0)

sys.argv = [sys.argv[0]] + _remaining_args

import hydra
import random
import numpy as np
import multiprocessing
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from datasets.caption.metrics import PTBTokenizer, Cider
from models.caption import Transformer
from models.caption.detector import build_detector
from tools.extract_features import extract_vis_features
from utils.cap_scheduler import CosineLRScheduler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from engine.caption_engine import *


def parse_args() -> DictConfig:
    """Parse configuration using Hydra and return it."""

    @hydra.main(config_path="configs/caption", config_name="coco_config")
    def _inner(cfg: DictConfig) -> DictConfig:
        return cfg

    return _inner()


def build_model(config: DictConfig, device: torch.device, gpu: int, rank: int) -> DDP:
    """Build detector and captioning model wrapped with DDP."""

    detector = build_detector(config).to(device)
    detector.load_state_dict(torch.load(config.model.detector.checkpoint)["model"], strict=False)

    model = Transformer(detector=detector, config=config).to(device)

    if config.optimizer.freezing_xe_epochs > 0:
        if getattr(config.optimizer, "freeze_backbone", False):
            for n, p in model.named_parameters():
                if "backbone" in n:
                    p.requires_grad = False

        if getattr(config.optimizer, "freeze_detector", False):
            for n, p in model.named_parameters():
                if "detector" in n:
                    p.requires_grad = False
        else:
            extract_vis_features(detector, config, device, rank)

    ddp_args = {"device_ids": [gpu]} if torch.cuda.is_available() else {}
    model = DDP(model, find_unused_parameters=True, broadcast_buffers=False, **ddp_args)
    return model


def load_data(
    config: DictConfig,
    model: DDP,
    device: torch.device,
    optimizers: dict,
    start_epoch: int,
) -> tuple:
    """Prepare dataloaders, text field and scheduler."""

    if start_epoch < config.optimizer.freezing_xe_epochs and not getattr(config.optimizer, "freeze_backbone", False):
        model.module.cached_features = True
        dataloaders, samplers = build_coco_dataloaders(config, mode="freezing", device=device)
    else:
        model.module.cached_features = False
        dataloaders, samplers = build_coco_dataloaders(config, mode="finetune", device=device)

    text_field = TextField(vocab_path=config.dataset.vocab_path)
    train_dataset = dataloaders["train"].dataset
    cider = Cider(PTBTokenizer.tokenize([e.text for e in train_dataset.examples]))
    tokenizer = multiprocessing.Pool(8)

    scheduler = CosineLRScheduler(
        optimizers["model"],
        num_epochs=config.optimizer.freezing_xe_epochs + config.optimizer.finetune_xe_epochs,
        num_its_per_epoch=len(dataloaders["train"]),
        init_lr=config.optimizer.xe_lr,
        min_lr=config.optimizer.min_lr,
        warmup_init_lr=config.optimizer.warmup_init_lr,
    )

    return dataloaders, samplers, text_field, cider, tokenizer, scheduler


def save_ckpt(model, optimizers, epoch: int, config: DictConfig, filename: str, scheduler) -> None:
    """Wrapper around engine.save_checkpoint for readability."""

    save_checkpoint(
        model,
        optimizers,
        epoch=epoch,
        scores=[],
        best_ciders=[0, 0],
        config=config,
        filename=filename,
        scheduler=scheduler,
    )


def train_loop(model: DDP, config: DictConfig, device: torch.device, rank: int, gpu: int) -> None:
    """Main training loop handling all training phases."""

    start_epoch = 0
    best_cider_val = 0.0
    best_cider_test = 0.0

    optimizers = build_optimizers(model, config, mode="xe")
    writer = SummaryWriter(log_dir="tensorboard") if rank in (0, 1) else None

    dataloaders, samplers, text_field, cider, tokenizer, scheduler = load_data(
        config, model, device, optimizers, start_epoch
    )

    fr_xe_epochs = config.optimizer.freezing_xe_epochs
    fr_sc_epochs = fr_xe_epochs + config.optimizer.freezing_sc_epochs
    ft_xe_epochs = fr_sc_epochs + config.optimizer.finetune_xe_epochs
    ft_sc_epochs = ft_xe_epochs + config.optimizer.finetune_sc_epochs
    total_epochs = ft_sc_epochs

    for epoch in range(max(0, start_epoch), total_epochs):
        if epoch < fr_xe_epochs:
            phase = "fr_xe"
        if fr_xe_epochs <= epoch < fr_sc_epochs:
            phase = "fr_sc"
        if fr_sc_epochs <= epoch < ft_xe_epochs:
            phase = "ft_xe"
        if ft_xe_epochs <= epoch < ft_sc_epochs:
            phase = "ft_sc"

        if (phase in ("ft_sc", "ft_xe")) and dataloaders["train"].dataset.image_field.use_hdf5_feat:
            model.module.cached_features = False
            dataloaders, samplers = build_coco_dataloaders(config, mode="finetune", device=device)

        if (phase in ("fr_sc", "ft_sc")) and optimizers["mode"] == "xe":
            optimizers = build_optimizers(model, config, mode="sc")

        if (phase in ("fr_xe", "ft_xe")) and optimizers["mode"] == "sc":
            optimizers = build_optimizers(model, config, mode="xe")

        print(f"Train: rank={rank}, epoch={epoch}, phase={phase}")
        if phase in ("fr_xe", "ft_xe"):
            train_res = train_xe(
                model,
                dataloaders,
                optimizers=optimizers,
                text_field=text_field,
                epoch=epoch,
                rank=rank,
                config=config,
                scheduler=scheduler,
                writer=writer,
            )
            samplers["train"].set_epoch(epoch)

        elif phase in ("fr_sc", "ft_sc"):
            checkpoint = torch.load("checkpoint_best_valid.pth", map_location="cpu")
            missing, unexpected = model.module.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Start self-critical optimization: missing={len(missing)}, unexpected={len(unexpected)}")
            train_res = train_sc(
                model,
                dataloaders,
                optimizers=optimizers,
                cider=cider,
                text_field=text_field,
                tokenizer_pool=tokenizer,
                device=device,
                epoch=epoch,
                rank=rank,
                config=config,
                writer=writer,
            )
            # build_coco_dataloaders (provided in the full dataset utilities) returns
            # a sampler keyed as "train_dict" for self-critical training.  When the
            # datasets submodule is absent, we cannot inspect this directly, but
            # the training functions expect this key for updating the epoch.
            samplers["train_dict"].set_epoch(epoch)

        if rank == 0:
            best_cider_val = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders["valid_dict"],
                text_field=text_field,
                epoch=epoch,
                split="valid",
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_val,
                which=phase,
                scheduler=scheduler,
            )

        if rank == 1:
            best_cider_test = evaluate_metrics(
                model,
                optimizers,
                dataloader=dataloaders["test_dict"],
                text_field=text_field,
                epoch=epoch,
                split="test",
                config=config,
                train_res=train_res,
                writer=writer,
                best_cider=best_cider_test,
                which=phase,
                scheduler=scheduler,
            )

        if rank == 0:
            save_ckpt(model, optimizers, epoch, config, f"checkpoint_{phase}.pth", scheduler)
            if epoch >= 15:
                save_ckpt(model, optimizers, epoch, config, f"checkpoint_{epoch}.pth", scheduler)

        torch.distributed.barrier()


def main(gpu: int, config: DictConfig) -> None:
    """Entry point for each spawned process."""

    torch.backends.cudnn.enabled = False
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, "env://", rank=rank, world_size=config.exp.world_size)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(gpu)
    else:
        device = torch.device("cpu")

    model = build_model(config, device, gpu, rank)
    train_loop(model, config, device, rank, gpu)


def run_main() -> None:
    config = parse_args()

    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()
