import torch
import torch.nn.functional as F
import lightning as L
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy
from transformers import CLIPModel

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