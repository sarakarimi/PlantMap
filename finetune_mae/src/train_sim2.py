import os
import lightning as L
import argparse

from torch.utils.data import DataLoader
from data.data_cont import GeneratorDataset
from models.model_byol import Model
from lightning.pytorch.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data2")
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_logger", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt")
    parser.add_argument("--vit", type=str, default="MAE") # options: MAE, clip, DINO
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    vit = args.vit
    use_logger = args.use_logger

    ckpt = L.pytorch.callbacks.ModelCheckpoint(
        monitor="valid/loss",
        dirpath="checkpoints",
        filename="finetune",
        save_top_k=1,
        enable_version_counter=False,  # creates model-v0, model-v1, etc.
    )
    callbacks = [ckpt]

    image_files = os.listdir(data_root)
    image_files = [os.path.join(data_root, file) for file in image_files]
    train_files = image_files[: int(0.8 * len(image_files))]
    val_files = image_files[int(0.8 * len(image_files)) :]


    lr = 0.3 * batch_size / 256

    train_generator = GeneratorDataset(train_files, vit, bs=batch_size)
    val_generator = GeneratorDataset(val_files, vit, bs=batch_size)

    train_loader = DataLoader(train_generator, batch_size=None)
    val_loader = DataLoader(val_generator, batch_size=None)

    model = Model(lr, checkpoint, vit)

    logger = WandbLogger(project="plant_map", save_dir="logs") if use_logger else None
    trainer = L.Trainer(max_epochs=num_epochs, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
