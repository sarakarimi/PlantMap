import torch
import random

from transformers import AutoImageProcessor
from torch.utils.data import IterableDataset


class NextImageBatchGenerator:
    def __init__(self, files: list[str], bs: int = 64) -> None:
        self._bs = bs
        self._image_size = 224
        self._files = files

        self._image_processor = AutoImageProcessor.from_pretrained(
            "facebook/vit-mae-base", use_fast=True
        )

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
            batch = self._image_processor(batch, return_tensors="pt")
            yield batch


class GeneratorDataset(IterableDataset):
    def __init__(self, files: list[str], bs: int = 64) -> None:
        super().__init__()
        self.generator_func = NextImageBatchGenerator(files, bs)

    def __iter__(self):
        return self.generator_func()



