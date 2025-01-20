import lightning as L
import torch

from transformers import ViTMAEForPreTraining


class Model(L.LightningModule):
    """
    Model for pretraining the Masked Autoencoder on the flower images, retrieved by SAM. 
    By masking the image, the model learns to predict the masked pixels.
    The trained model can be used for fine-tuning on unsupervised classification tasks. 
    """
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
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.forward(batch)
        self.log("valid/loss", outputs.loss, on_epoch=True, prog_bar=True)
        return outputs.loss
