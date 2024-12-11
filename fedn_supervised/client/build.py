from omegaconf import DictConfig
import torch
import torch.nn as nn
import os
import sys
import random
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from data import ImageTextDataset, collate_fn
from lightning.pytorch.loggers import CSVLogger

from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from transformers import CLIPModel, CLIPProcessor
from fedn.utils.helpers.helpers import save_metadata

from model import CLIPClassifier, CLIPContrastiveClassifier, CLIPLightningWithContrastive
from data import preprocess_dataset
from utils import get_dataset_indices, get_hydra_conf


def main(cfg: DictConfig) -> None:
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


    # Load the CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"

    dataset_name = cfg.model.dataset_name
    # ds = load_dataset("xavantex/EindhovenWildflower")
    # ds = load_dataset("sarakarimi30/PlantMap")

    ds = load_dataset(dataset_name)
    ds_size = len(ds["train"])
    all_indices = list(range(ds_size))


    dataset = ds["train"]
    # TODO: Hacky trick to get all categories without changing the code too much -> bad code design
    _, _, categories = preprocess_dataset(dataset, all_indices)

    num_classes = len(categories)

    # Either finetune base model or load a pretrained model
    if cfg.model.finetune_checkpoint:
        checkpoint_path = hf_hub_download(
            repo_id="SIAndersson/PlantMapCheckpoints",
            filename=cfg.model.pretrained_checkpoint,
        )
        # Pretrained model was contrastive
        if pretrained_str == "contrastive":
            clip_model = CLIPModel.from_pretrained(model_name).train()
            base_model = CLIPLightningWithContrastive.load_from_checkpoint(
                checkpoint_path, clip_model=clip_model
            )
            model = CLIPContrastiveClassifier(
                base_model,
                num_classes,
                lr=cfg.training.learning_rate,
                optimizer_type=cfg.training.optimizer,
                dropout_rate=cfg.model.dropout,
            )
        # Pretrained model was categorical
        else:
            model = CLIPClassifier.load_from_checkpoint(checkpoint_path)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            model.accuracy = MulticlassAccuracy(num_classes=num_classes)
            model.lr = cfg.training.learning_rate
            model.optimizer_type = cfg.training.optimizer

            # Reset classifier head weights since outputs have changed
            nn.init.kaiming_normal_(model.classifier.weight)
            nn.init.zeros_(model.classifier.bias)
    # Using base CLIP model
    else:
        model = CLIPClassifier(
            model_name,
            num_classes,
            optimizer_type=cfg.training.optimizer,
            lr=cfg.training.learning_rate,
        )

    # Freeze the feature extractor
    for param in model.clip_model.parameters():
        param.requires_grad = False

    trainer = Trainer(
        logger=None,
        callbacks=[],
    )

    trainer.strategy.connect(model)
    trainer.save_checkpoint("seed.npz")




if __name__ == "__main__":
    # Set high precision for matrix multiplication (for tensor cores)
    torch.set_float32_matmul_precision("high")
    cfg = get_hydra_conf()
    main(cfg)
