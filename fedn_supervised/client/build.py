from omegaconf import DictConfig
import torch

from datasets import load_dataset
from lightning.pytorch import Trainer


from data import preprocess_dataset
from utils import (
    get_hydra_conf,
    load_model_from_cfg,
    save_parameters,
)


def main(cfg: DictConfig) -> None:
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

    model = load_model_from_cfg(num_classes, cfg)

    # NOTE: Freezing here allows to store subset of weights
    for param in model.clip_model.parameters():
        param.requires_grad = False

    save_parameters(model, "seed.npz")


if __name__ == "__main__":
    # Set high precision for matrix multiplication (for tensor cores)
    torch.set_float32_matmul_precision("high")
    cfg = get_hydra_conf()
    main(cfg)
