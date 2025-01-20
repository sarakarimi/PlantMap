import lightning as L
import torch

from utils.similarity_loss import SimilarityLoss
from models.model_pretrain import Model as PretrainedModel
from transformers import CLIPVisionModel
from transformers import Dinov2Model


class Model(L.LightningModule):
    """
    Using the contrastive learning method for unsupervised learning.
    See https://arxiv.org/abs/2006.10029 for more information.
    """
    def __init__(self, lr: float, checkpoint: str | None, vit: str = "MAE") -> None:
        super().__init__()
        self._lr = lr
        if vit == "MAE":
            checkpoint = "facebook/vit-mae-base" if checkpoint is None else checkpoint
            self._encoder = PretrainedModel.load_from_checkpoint(checkpoint)._model.vit.train()
        elif vit == "CLIP":
            if checkpoint is not None:
                print("WARNING: Ignoring checkpoint for CLIP model")
            checkpoint = "openai/clip-vit-base-patch32" 
            self._encoder = CLIPVisionModel.from_pretrained(checkpoint).train()
        elif vit == "DINO":
            if checkpoint is not None:
                print("WARNING: Ignoring checkpoint for DINO model")
            checkpoint = "facebook/dinov2-base"
            self._encoder = Dinov2Model.from_pretrained(checkpoint).train()
        else:
            raise ValueError(f"Unknown model type: {vit}")
        self._similarity_loss = SimilarityLoss()

    def forward(self, pixel_values):
        return self._encoder(pixel_values).last_hidden_state[:, 0]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Input: 2 btaches of the same images with different augmentations.
        Output: The loss of the contrastive learning. 
        """
        batch1, batch2 = batch
        outputs1 = self.forward(batch1.pixel_values)
        outputs2 = self.forward(batch2.pixel_values)
        loss = self.__compute_loss(outputs1, outputs2)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def __compute_loss(self, outputs1, outputs2):
        loss = self._similarity_loss(outputs1, outputs2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self._encoder.parameters(), "lr": 1e-4},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        batch1, batch2 = batch
        outputs1 = self.forward(batch1.pixel_values)
        outputs2 = self.forward(batch2.pixel_values)
        loss = self.__compute_loss(outputs1, outputs2)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        return loss
