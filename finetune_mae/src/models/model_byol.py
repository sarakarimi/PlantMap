import lightning as L
import torch
import copy
import torch.nn.functional as F
import torch.nn as nn

from utils.similarity_loss import SimilarityLoss
from models.model_pretrain import Model as PretrainedModel
from transformers import CLIPVisionModel
from transformers import Dinov2Model


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Model(L.LightningModule):
    def __init__(self, checkpoint: str | None, vit: str = "MAE") -> None:
        super().__init__()
        if vit == "MAE":
            checkpoint = "facebook/vit-mae-base" if checkpoint is None else checkpoint
            self._encoder = PretrainedModel.load_from_checkpoint(checkpoint)._model.vit
        elif vit == "CLIP":
            if checkpoint is not None:
                print("WARNING: Ignoring checkpoint for CLIP model")
            checkpoint = "openai/clip-vit-base-patch32"
            self._encoder = CLIPVisionModel.from_pretrained(checkpoint)
        elif vit == "DINO":
            if checkpoint is not None:
                print("WARNING: Ignoring checkpoint for DINO model")
            checkpoint = "facebook/dinov2-base"
            self._encoder = Dinov2Model.from_pretrained(checkpoint)
        else:
            raise ValueError(f"Unknown model type: {vit}")
        self._encoder2 = copy.deepcopy(self._encoder)  # target network
        self._mlp = MLP(768, 1000, 768)
        self.ema_decay = 0.99
        self._similarity_loss = SimilarityLoss()

    def forward(self, pixel_values):
        return self._encoder(pixel_values).last_hidden_state[:, 0]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        batch1, batch2 = batch
        loss = self.__shared_step(batch1, batch2)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def __shared_step(self, batch1, batch2):
        outputs1_online = self._mlp(
            self._encoder(batch1.pixel_values).last_hidden_state[:, 0]
        )
        outputs2_online = self._mlp(
            self._encoder2(batch2.pixel_values).last_hidden_state[:, 0]
        )
        with torch.no_grad():
            outputs1_target = self._encoder(batch1.pixel_values).last_hidden_state[:, 0]
            outputs2_target = self._encoder2(batch2.pixel_values).last_hidden_state[
                :, 0
            ]

        loss1 = self.__byol_loss(outputs1_online, outputs2_target)
        loss2 = self.__byol_loss(outputs2_online, outputs1_target)

        return loss1 + loss2

    def __byol_loss(self, pred, target):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()

    def __compute_loss(self, outputs1, outputs2):
        loss = self._similarity_loss(outputs1, outputs2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self._encoder.parameters(), "lr": 1e-3},
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        batch1, batch2 = batch
        outputs1 = self.forward(batch1.pixel_values)
        outputs2 = self.forward(batch2.pixel_values)
        loss = self.__compute_loss(outputs1, outputs2)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def __update_target_network(self, ema_decay):
        for online_params, target_params in zip(
            self._encoder.parameters(), self._encoder2.parameters()
        ):
            target_params.data = (
                ema_decay * target_params.data + (1 - ema_decay) * online_params.data
            )

    def on_train_epoch_end(self):
        self.__update_target_network(self.ema_decay)
