import torch
import random
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoImageProcessor
from torch.utils.data import IterableDataset
from torchvision import transforms


class NextImageBatchGenerator:
    def __init__(self, files: list[str], bs: int = 64) -> None:
        self._bs = bs
        self._image_size = 224
        self._files = files

        self._image_processor = AutoImageProcessor.from_pretrained(
            "facebook/vit-mae-base", use_fast=True
        )

        self._augmentations = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self):
        random.shuffle(self._files)
        batch1 = []
        batch2 = []
        for img_path in self._files:
            images = torch.load(img_path)
            random.shuffle(images)
            for img in images:
                img = Image.fromarray(img)
                img1 = self._augmentations(img)
                img2 = self._augmentations(img)
                batch1.append(img1)
                batch2.append(img2)
                if len(batch1) == self._bs:
                    assert len(batch1) == len(batch2)
                    batch1 = self._image_processor(batch1, return_tensors="pt")
                    batch2 = self._image_processor(batch2, return_tensors="pt")
                    yield batch1, batch2
                    batch1 = []
                    batch2 = []
        if batch1:
            assert len(batch1) == len(batch2)
            batch1 = self._image_processor(batch1, return_tensors="pt")
            batch2 = self._image_processor(batch2, return_tensors="pt")
            yield batch1, batch2

class GeneratorDataset(IterableDataset):
    def __init__(self, files: list[str], bs: int = 64) -> None:
        super().__init__()
        self.generator_func = NextImageBatchGenerator(files, bs)

    def __iter__(self):
        return self.generator_func()