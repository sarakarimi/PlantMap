import torch
import os
import random

from transformers import AutoImageProcessor
from torchvision.transforms import ToTensor, Normalize, Compose, Resize


class NextImageBatchGenerator:
    def __init__(self, bs: int = 64) -> None:
        self._bs = bs
        self._image_size = 224
        self._files = [f for f in os.listdir("../data2")]
        self._files = [f"../data2/{f}" for f in self._files if f.endswith(".pt")]

        self._transform = Compose(
            [
                ToTensor(),
                Resize((self._image_size, self._image_size)),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self._image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")


    def __call__(self):
        random.shuffle(self._files)
        batch = []
        for img_path in self._files:
            images = torch.load(img_path)
            random.shuffle(images)
            for img in images:
                batch.append(img)
                if len(batch) == self._bs:
                    batch = self._image_processor(batch, return_tensors="pt")
                    yield batch
                    batch = []
        if batch:
            yield torch.stack(batch)
