from omegaconf import DictConfig
import torch
import sys
from datasets import load_dataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
)

from data import ImageTextDataset, collate_fn
from lightning.pytorch.loggers import CSVLogger

from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from fedn.utils.helpers.helpers import save_metadata

from data import preprocess_dataset
from utils import get_dataset_indices, get_hydra_conf
from utils import load_model_from_cfg, load_parameters, save_parameters


def main(in_model_path: str, out_model_path: str, cfg: DictConfig) -> None:
    """
    Train class for federated learning.
    The data is expeceted to be on the device. 
    Using a particular seed, the training data is split into 6 parts 
    while the validation data is the same for all clients.
    The training process is identical to the sam-clip training process.
    """
    hyper_params = {
        "model": "CLIP",
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "optimizer": cfg.training.optimizer,
        "dataset_name": cfg.model.dataset_name,
        "dropout": cfg.model.dropout,
        "finetune_checkpoint": cfg.model.finetune_checkpoint,
        "pretrained checkpoint": cfg.model.pretrained_checkpoint,
    }

    pretrained_str = (
        "contrastive"
        if "val_loss" in cfg.model.pretrained_checkpoint
        else "categorical"
    )
    model_str = f"{pretrained_str if cfg.model.finetune_checkpoint else "base"}CLIP_dropout_{cfg.model.dropout}_optim_{cfg.training.optimizer}_batch_{cfg.training.batch_size}_lr_{cfg.training.learning_rate}"

    logger = CSVLogger("logs", name=model_str)

    logger.log_hyperparams(hyper_params)

    # Load the CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    dataset_name = cfg.model.dataset_name
    # ds = load_dataset("xavantex/EindhovenWildflower")
    # ds = load_dataset("sarakarimi30/PlantMap")

    ds = load_dataset(dataset_name)
    ds_size = len(ds["train"])
    all_indices = list(range(ds_size))
    split_idx = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split_idx]
    # val_indices = all_indices[split_idx:]

    train_indices = get_dataset_indices(train_indices)

    dataset = ds["train"]
    # TODO: Hacky trick to get all categories without changing the code too much -> bad code design
    _, _, categories = preprocess_dataset(dataset, all_indices)
    categories = sorted(categories)

    train_images, train_labels, _ = preprocess_dataset(dataset, train_indices)
    # val_images, val_labels, _ = preprocess_dataset(dataset, val_indices)

    num_classes = len(categories)

    train_dataset = ImageTextDataset(
        train_images, train_labels, categories, clip_processor
    )

    # NOTE: val dataset only used in validation script
    # val_dataset = ImageTextDataset(val_images, val_labels, categories, clip_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model = load_model_from_cfg(num_classes, cfg)

    for param in model.clip_model.parameters():
        param.requires_grad = False

    model = load_parameters(model, in_model_path)


    trainer_params = {
        "logger": [logger],
        "max_epochs": hyper_params["epochs"], 
        "accelerator": "auto",
        "log_every_n_steps": 10,
        "callbacks": [lr_monitor],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        trainer_params["devices"] = (
            cfg.training.num_gpus
            if cfg.training.num_gpus > 1
            else [int(cfg.training.device[-1])]
        )

    trainer = Trainer(**trainer_params)

    trainer.fit(model, train_dataloaders=train_loader)
    save_parameters(model, out_model_path)

    # Get the best epoch and corresponding accuracy

    metadata = {
        # num_examples are mandatory
        "num_examples": len(train_indices),
        "batch_size": hyper_params["batch_size"],
        "epochs": hyper_params["epochs"],
        "lr": hyper_params["learning_rate"],
    }

    # FedN metadata
    save_metadata(metadata, out_model_path)



if __name__ == "__main__":
    # Set high precision for matrix multiplication (for tensor cores)
    torch.set_float32_matmul_precision("high")
    in_model_path = sys.argv[1]
    out_model_path = sys.argv[2]
    cfg = get_hydra_conf()
    main(in_model_path, out_model_path, cfg)
