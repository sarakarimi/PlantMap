import torch
import os
import lightning as L
import argparse

from torch.utils.data import DataLoader
from data import GeneratorDataset
from model import Model
from lightning.pytorch.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data2")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4
    )  # TODO: Not used yet
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    image_files = os.listdir(data_root)
    image_files = [os.path.join(data_root, file) for file in image_files]
    train_files = image_files[: int(0.8 * len(image_files))]
    val_files = image_files[int(0.8 * len(image_files)) :]

    train_generator = GeneratorDataset(train_files, bs=batch_size)
    val_generator = GeneratorDataset(val_files, bs=batch_size)

    train_loader = DataLoader(train_generator, batch_size=None)
    val_loader = DataLoader(val_generator, batch_size=None)

    model = Model()

    logger = WandbLogger(project="plant_map")
    trainer = L.Trainer(max_epochs=num_epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
