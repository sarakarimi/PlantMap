import lightning as L
import torch

from utils.similarity_loss import SimilarityLoss
from model_pretrain import Model as PretrainedModel


class Model(L.LightningModule):
    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        self._encoder = PretrainedModel.load_from_checkpoint(checkpoint)._model.vit
        # self._encoder = PretrainedModel()._model.vit
        self._similarity_loss = SimilarityLoss()

    def forward(self, pixel_values):
        return self._encoder(pixel_values).last_hidden_state[:, 0]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
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
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        batch1, batch2 = batch
        outputs1 = self.forward(batch1.pixel_values)
        outputs2 = self.forward(batch2.pixel_values)
        loss = self.__compute_loss(outputs1, outputs2)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        return loss
