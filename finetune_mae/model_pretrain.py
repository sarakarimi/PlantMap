import lightning as L
import torch

from transformers import ViTMAEForPreTraining


class Model(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

    def forward(self, inputs):
        return self._model(**inputs)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        self.log("train/loss", outputs.loss, on_step=True, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.forward(batch)
        self.log("valid/loss", outputs.loss, on_epoch=True, prog_bar=True)
        return outputs.loss
