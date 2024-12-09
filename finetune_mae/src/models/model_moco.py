import torch
import torch.nn as nn
import lightning as L
from models.model_pretrain import Model as PretrainedModel
from transformers import CLIPVisionModel
from transformers import Dinov2Model
from transformers import ViTMAEModel

from utils.similarity_loss import SimilarityLoss

class Model(L.LightningModule):
    def __init__(self, checkpoint: str, vit: str ="MAE"):
        super().__init__()
        
        device = "cuda" if  torch.cuda.is_available() else "cpu"
        if vit == "MAE":
            if checkpoint is None:
                self._encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base").train().to(device)
            else: 
                self._encoder = PretrainedModel.load_from_checkpoint(checkpoint)._model.vit.train().to(device)
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
        self.projection_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self._similarity_loss = SimilarityLoss()
        
        self.queue_size = 1024
        self.queue = torch.randn(128, self.queue_size)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = 0  
        
        self.temperature = 0.07
    
    def forward(self, x):
        features = self._encoder(x.pixel_values).last_hidden_state[:, 0]  # Use [CLS] token embedding
        return self.projection_head(features)
    
    def momentum_update(self):
        for param_q, param_k in zip(self.parameters(), self.queue.parameters()):
            param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)
    
    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch  # Two augmented views of the same images
        
        q1, q2 = self(batch1), self(batch2)
        
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q1, q2]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q1, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        self._dequeue_and_enqueue(q2)
        
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        batch1, batch2 = batch
        outputs1 = self.forward(batch1)
        outputs2 = self.forward(batch2)
        loss = self.__compute_loss(outputs1, outputs2)
        self.log("valid/loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        
        # Replace the keys at ptr (FIFO)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        
        self.queue_ptr = ptr
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    
    def __compute_loss(self, outputs1, outputs2):
        loss = self._similarity_loss(outputs1, outputs2)
        return loss