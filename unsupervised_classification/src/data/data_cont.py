import torch
import random

from PIL import Image
from transformers import AutoImageProcessor, CLIPProcessor
from torch.utils.data import IterableDataset
from torchvision import transforms
from utils.no_flower_filter import no_flower_filter


class NextImageBatchGenerator:
    """
    A RAM efficient generator for the image data.
    Since having all the images in memory at once is not feasible, this generator
    yields batches of images from the files.
    The data is loaded from preprocessed files containing torch tensors of the images
    retrieved from SAM and are augmented before being processed serving as the base
    for contrastive loss training.
    """
    def __init__(self, files: list[str], vit: str, bs: int = 64) -> None:
        self._bs = bs
        self._image_size = 224
        self._files = files

        if vit == "MAE":
            self._image_processor = AutoImageProcessor.from_pretrained(
                "facebook/vit-mae-base", use_fast=True
            )
        elif vit == "CLIP":
            self._image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif vit == "DINO":
            self._image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        else:
            raise ValueError(f"Unknown model type: {vit}")

        self._augmentations = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(3),
                transforms.ToTensor(),
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
                if no_flower_filter(img):
                    pass # TODO: Exchange with continue
                img = Image.fromarray(img)
                img1 = self._augmentations(img)
                img2 = self._augmentations(img)
                batch1.append(img1)
                batch2.append(img2)
                if len(batch1) == self._bs:
                    assert len(batch1) == len(batch2)
                    batch1 = self._image_processor(images=batch1, return_tensors="pt")
                    batch2 = self._image_processor(images=batch2, return_tensors="pt")
                    yield batch1, batch2
                    batch1 = []
                    batch2 = []
        if batch1:
            assert len(batch1) == len(batch2)
            batch1 = self._image_processor(batch1, return_tensors="pt")
            batch2 = self._image_processor(batch2, return_tensors="pt")
            yield batch1, batch2

class GeneratorDataset(IterableDataset):
    """
    This generator iterates through all batches until the end of the dataset.
    """
    def __init__(self, files: list[str], vit: str, bs: int = 64) -> None:
        super().__init__()
        self.generator_func = NextImageBatchGenerator(files, vit, bs)

    def __iter__(self):
        return self.generator_func()