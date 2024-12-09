import hydra
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class ContrastiveLossWithTemperature(torch.nn.Module):
    """
    ContrastiveLossWithTemperature is a custom loss function for contrastive learning that includes a learnable temperature parameter.

    Args:
        initial_temperature (float, optional): The initial value for the temperature parameter. Default is 1.0.

    Methods:
        forward(image_embeds, text_embeds):
            Computes the contrastive loss between image and text embeddings.

            Args:
                image_embeds (torch.Tensor): The embeddings for the images.
                text_embeds (torch.Tensor): The embeddings for the texts.

            Returns:
                tuple: A tuple containing the cross-entropy loss for images and texts.
    """

    def __init__(self, initial_temperature=1.0):
        super().__init__()
        # Initialize temperature as a learnable parameter
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute logits and scale by temperature
        logits_per_image = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        logits_per_text = logits_per_image.T

        # Labels for positive pairs
        labels = torch.arange(len(image_embeds), device=image_embeds.device)

        # Cross-entropy loss for both directions
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        # Average loss
        return loss_i, loss_t


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


def preprocess_dataset(dataset):
    """
    Preprocesses a dataset of images by cropping them based on bounding boxes and extracting labels.

    Args:
        dataset (list of dict): A list of dictionaries where each dictionary represents an image file.
            Each dictionary should contain the following keys:
            - "image": The image data.
            - "labels": A list of labels corresponding to objects in the image.
            - "boxes" or "bboxes": A list of bounding boxes for the objects in the image.

    Returns:
        tuple: A tuple containing:
            - cropped_images (list): A list of cropped images based on the bounding boxes.
            - labels (list): A list of labels corresponding to the cropped images.
            - categories (list): A list of unique categories found in the dataset.
    """
    cropped_images = []
    labels = []
    categories = set()
    box_str = "boxes" if "boxes" in dataset[0] else "bboxes"

    for file in tqdm(dataset, desc="Preprocessing dataset"):
        for label, box in zip(file["labels"], file[box_str]):
            cropped_images.append(preprocess_image(file["image"], box))
            categories.add(label)
            labels.append(label)

    return cropped_images, labels, list(categories)


class CLIPClassifier(L.LightningModule):
    """
    A PyTorch Lightning Module for fine-tuning a CLIP model for classification tasks.

    Args:
        model_name (str): The name of the pre-trained CLIP model to use.
        num_classes (int): The number of output classes for the classifier.
        optimizer_type (str): The type of optimizer to use ('adamw', 'sgd', or 'adadelta'). Default is 'sgd'.
        freeze_grad (bool, optional): Whether to freeze the gradients of the CLIP model. Default is False.
        lr (float, optional): The learning rate for the optimizer. Default is 1e-3.

    Attributes:
        clip_model (CLIPModel): The pre-trained CLIP model.
        classifier (nn.Linear): The linear layer used for classification.
        criterion (nn.CrossEntropyLoss): The loss function.
        lr (float): The learning rate for the optimizer.
        accuracy (MulticlassAccuracy): The accuracy metric for validation.
        optimizer_type (str): The type of optimizer to use.

    Methods:
        forward(pixel_values):
            Forward pass through the model.

        training_step(batch, batch_idx):
            Training step for a single batch.

        validation_step(batch, batch_idx):
            Validation step for a single batch.

        on_validation_epoch_end():
            Compute and log validation accuracy at the end of an epoch.

        configure_optimizers():
            Configure the optimizer and learning rate scheduler.
    """

    def __init__(
        self, model_name, num_classes, optimizer_type="sgd", freeze_grad=False, lr=1e-3
    ):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name).train()
        self.classifier = nn.Linear(self.clip_model.config.projection_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_grad = freeze_grad
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.optimizer_type = optimizer_type.lower()

        self.save_hyperparameters(ignore=["clip_model"])

    def forward(self, pixel_values):
        # Extract image embeddings
        image_embeds = self.clip_model.get_image_features(pixel_values)
        # Pass through classifier head
        logits = self.classifier(image_embeds)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = (
            batch["pixel_values"],
            batch["labels"],
        )

        outputs = self(images)

        loss = self.criterion(outputs, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["pixel_values"], batch["labels"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.accuracy.update(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute() * 100
        self.log("val_accuracy", accuracy, sync_dist=True)
        self.accuracy.reset()

    def configure_optimizers(self):
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        elif self.optimizer_type.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr * 100)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class CLIPContrastiveClassifier(L.LightningModule):
    """
    A PyTorch Lightning module that integrates a pretrained contrastive CLIP model with a classifier head for fine-tuning.

    Args:
        clip_model (nn.Module): The pre-trained CLIP model to be used.
        num_classes (int): The number of output classes for the classifier.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        optimizer_type (str, optional): Type of optimizer to use. Options are "adamw", "sgd", "adadelta". Default is "adadelta".
        dropout_rate (float, optional): Dropout rate to be applied before the classifier head. Default is 0.2.

    Attributes:
        clip_model (nn.Module): The pre-trained CLIP model.
        dropout (nn.Dropout): Dropout layer.
        classifier (nn.Linear): Linear layer for classification.
        lr (float): Learning rate.
        accuracy (MulticlassAccuracy): Metric for accuracy.
        optimizer_type (str): Type of optimizer.
        criterion (nn.CrossEntropyLoss): Loss function.

    Methods:
        forward(pixel_values, input_ids, attention_mask):
            Forward pass through the model.

        training_step(batch, batch_idx):
            Training step for a single batch.

        validation_step(batch, batch_idx):
            Validation step for a single batch.

        on_validation_epoch_end():
            Actions to perform at the end of a validation epoch.

        configure_optimizers():
            Configures the optimizer and learning rate scheduler.
    """

    def __init__(
        self,
        clip_model,
        num_classes,
        lr=1e-4,
        optimizer_type="adadelta",
        dropout_rate=0.2,
    ):
        super().__init__()
        self.clip_model = clip_model  # Use the loaded CLIP model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            self.clip_model.clip_model.config.projection_dim, num_classes
        )
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.lr = lr
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.optimizer_type = optimizer_type.lower()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pixel_values, input_ids, attention_mask):
        # Extract embeddings
        image_embeds, _ = self.clip_model(pixel_values, input_ids, attention_mask)
        # Apply dropout
        dropped_out_embeds = self.dropout(image_embeds)
        # Pass through the classification head
        logits = self.classifier(dropped_out_embeds)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values, input_ids, attention_mask, labels = (
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self.forward(pixel_values, input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, attention_mask, labels = (
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        outputs = self(pixel_values, input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.accuracy.update(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        accuracy = self.accuracy.compute() * 100
        self.log("val_accuracy", accuracy, sync_dist=True)
        self.accuracy.reset()

    def configure_optimizers(self):
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        elif self.optimizer_type.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr * 100)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class CLIPLightningWithContrastive(L.LightningModule):
    """
    PyTorch Lightning module for training a CLIP model with contrastive loss.

    Args:
        clip_model (nn.Module): The CLIP model to be trained.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        temperature (float, optional): Temperature parameter for the contrastive loss. Default is 0.07.
        optimizer_type (str, optional): Type of optimizer to use. Options are "adamw", "sgd", and "adadelta". Default is "adadelta".

    Methods:
        forward(pixel_values, input_ids, attention_mask):
            Computes image and text embeddings using the CLIP model.

        recall_at_k(image_embeds, text_embeds, labels, k=5):
            Computes the recall at k metric for the given embeddings and labels.

        mean_rank(image_embeds, text_embeds, labels):
            Computes the mean rank of the correct labels for the given embeddings.

        training_step(batch, batch_idx):
            Defines the training step for the model.

        validation_step(batch, batch_idx):
            Defines the validation step for the model.

        configure_optimizers():
            Configures the optimizer and learning rate scheduler for the model.
    """

    def __init__(
        self, clip_model, lr=1e-4, temperature=0.07, optimizer_type="adadelta"
    ):
        super().__init__()
        self.clip_model = clip_model
        self.lr = lr
        self.temperature = temperature
        self.criterion = ContrastiveLossWithTemperature(temperature)
        self.optimizer_type = optimizer_type.lower()

    def forward(self, pixel_values, input_ids, attention_mask):
        # Compute image and text embeddings
        image_embeds = self.clip_model.get_image_features(pixel_values)
        text_embeds = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return image_embeds, text_embeds

    def recall_at_k(self, image_embeds, text_embeds, labels, k=5):
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarities = torch.matmul(
            image_embeds, text_embeds.T
        )  # Compute cosine similarity
        ranks = similarities.argsort(dim=-1, descending=True)  # Rank text embeddings

        # Check if the correct label is in the top-K for each image
        top_k = ranks[:, :k]
        correct = (top_k == labels.unsqueeze(1)).any(dim=1)
        return correct.float().mean().item()  # Fraction of correct matches

    def mean_rank(self, image_embeds, text_embeds, labels):
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarities = torch.matmul(image_embeds, text_embeds.T)
        ranks = similarities.argsort(dim=-1, descending=True)  # Rank text embeddings

        # Find the rank of the correct label for each image
        rank_positions = (ranks == labels.unsqueeze(1)).nonzero()[:, 1]
        return rank_positions.float().mean().item()  # Average rank

    def training_step(self, batch, batch_idx):
        image_embeds, text_embeds = self(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
        )
        loss_i, loss_t = self.criterion(image_embeds, text_embeds)
        loss = (loss_i + loss_t) / 2
        self.log("train_loss", loss)
        self.log("train_loss_image", loss_i)
        self.log("train_loss_text", loss_t)
        return loss

    def validation_step(self, batch, batch_idx):
        image_embeds, text_embeds = self(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
        )
        loss_i, loss_t = self.criterion(image_embeds, text_embeds)
        loss = (loss_i + loss_t) / 2
        k_recall = self.recall_at_k(image_embeds, text_embeds, batch["labels"])
        mean_rank = self.mean_rank(image_embeds, text_embeds, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_image", loss_i)
        self.log("val_loss_text", loss_t)
        self.log("val_recall@5", k_recall, prog_bar=True)
        self.log("val_mean_rank", mean_rank, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        elif self.optimizer_type.lower() == "adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr * 100)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, threshold=0.01
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class ImageTextDataset(Dataset):
    """
    A custom dataset class for handling image and text data for CLIP model training.

    Args:
        images (list): A list of preprocessed images.
        text_labels (list): A list of text labels corresponding to the images.
        categories (list): A list of unique categories for the labels.
        processor (CLIPProcessor): A processor for preparing the data for the CLIP model.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Returns a single sample from the dataset at the given index.
    """

    def __init__(self, images, text_labels, categories, processor):
        self.images = images
        self.text_labels = text_labels
        self.processor = processor
        self.categories = categories

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text_label = self.text_labels[idx]
        label = self.categories.index(text_label)
        processed = self.processor(
            text=text_label, images=image, return_tensors="pt", padding=True
        )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": label,
        }


def collate_fn(batch):
    """
    Collates a batch of data for a model that processes both images and text.

    Args:
        batch (list of dict): A list where each element is a dictionary containing:
            - "pixel_values" (torch.Tensor): The tensor representing image pixel values.
            - "input_ids" (torch.Tensor): The tensor of input token IDs for text.
            - "attention_mask" (torch.Tensor): The tensor of attention masks for text.
            - "labels" (int or torch.Tensor): The label associated with the data.

    Returns:
        dict: A dictionary with the following keys:
            - "pixel_values" (torch.Tensor): A stacked tensor of image pixel values.
            - "input_ids" (torch.Tensor): A padded tensor of input token IDs.
            - "attention_mask" (torch.Tensor): A padded tensor of attention masks.
            - "labels" (torch.Tensor): A tensor of labels.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,  # Padding token ID
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.tensor([item["labels"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@hydra.main(config_path="config", config_name="CLIP", version_base=None)
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
    dataset = ds["train"]
    cropped_images, labels, categories = preprocess_dataset(dataset)

    num_classes = len(categories)

    train_images, val_images, train_labels, val_labels = train_test_split(
        cropped_images, labels, test_size=0.2, random_state=999, shuffle=True
    )

    train_dataset = ImageTextDataset(
        train_images, train_labels, categories, clip_processor
    )
    val_dataset = ImageTextDataset(val_images, val_labels, categories, clip_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_fn
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints",
        filename=model_str + "_best_model-{epoch:02d}-{val_accuracy:.2f}",
        monitor="val_accuracy",
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", min_delta=0.001, patience=15, verbose=False, mode="max"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

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
        logger=[logger],
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=cfg.training.num_gpus
        if cfg.training.num_gpus > 1
        else [int(cfg.training.device[-1])],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Get the best epoch and corresponding accuracy
    best_path = checkpoint_callback.best_model_path
    best_val_accuracy = checkpoint_callback.best_model_score

    print(f"Best model path: {best_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    # Set high precision for matrix multiplication (for tensor cores)
    torch.set_float32_matmul_precision("high")
    main()
