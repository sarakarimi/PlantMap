import os
import random
import torch.nn as nn
import lightning as L
import collections
import torch

from hydra import initialize, compose
from fedn.utils.helpers.helpers import get_helper
from omegaconf import DictConfig
from huggingface_hub import hf_hub_download
from transformers import CLIPModel
from model import (
    CLIPClassifier,
    CLIPContrastiveClassifier,
    CLIPLightningWithContrastive,
)
from torchmetrics.classification import MulticlassAccuracy

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def get_dataset_indices(ds_indices: int) -> list[int]:
    """
    Slurm sets env variables NUM_AGENTS and AGENT_ID to distribute the dataset among agents.
    This function retrieves the indices for the dataset for the current agent.
    All indices sets are disjoint and randomized.
    """
    num_agents = os.environ["NUM_AGENTS"]
    print(num_agents)
    num_agents = int(num_agents)
    agent_id = os.environ["AGENT_ID"]
    agent_id = int(agent_id)

    random.shuffle(ds_indices)
    start_idx = agent_id * int(len(ds_indices) / int(num_agents))
    end_idx = start_idx + int(len(ds_indices) / int(num_agents))
    print(start_idx, end_idx, len(ds_indices))
    return ds_indices[start_idx:end_idx]


def get_hydra_conf():
    """
    Workaround for Hydra initialization in the training script while preserving the first arguments for fedN.
    """
    with open("overrides.txt", "r") as f:
        overrides = f.read().split("\n")
    if overrides[-1] == "":
        overrides = overrides[:-1]

    with initialize(config_path="config", job_name="CLIP", version_base="1.1"):
        cfg = compose(config_name="CLIP", overrides=overrides)
    return cfg


def preprocess_image(image, box):
    """
    Preprocesses the given image by cropping it to the specified bounding box and converting it to RGB format.

    Args:
        image (PIL.Image.Image): The input image to be preprocessed.
        box (tuple): A tuple of four integers (left, upper, right, lower) defining the bounding box to crop the image.

    Returns:
        PIL.Image.Image: The preprocessed image in RGB format.
    """
    image = image.crop((box[0], box[1], box[2], box[3]))
    image = image.convert("RGB")
    return image


def load_model_from_cfg(num_classes, cfg: DictConfig) -> L.LightningModule:
    pretrained_str = (
        "contrastive"
        if "val_loss" in cfg.model.pretrained_checkpoint
        else "categorical"
    )
    model_name = "openai/clip-vit-base-patch32"
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
    return model


def save_parameters(model: L.LightningModule, out_path: str) -> None:
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [
        val.cpu().numpy() for _, val in model.state_dict().items() if val.requires_grad
    ]
    helper.save(parameters_np, out_path)


def load_parameters(model: L.LightningModule, model_path: str) -> L.LightningModule:
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict(
        {key: torch.tensor(x) for key, x in params_dict}
    )
    model.load_state_dict(
        state_dict, strict=False
    )  # NOTE: false bcs state_dict has only classifier
    return model
