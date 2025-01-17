import torch
import numpy as np


def no_flower_filter(img_array: np.array) -> bool:
    """
    Filters with high accuracy flower images with a high ratio of green pixels.
    The idea is that SAM images with a high ratio of green pixels are likely to be
    grass or other non-flower images.
    """
    img = torch.tensor(img_array, dtype=torch.float32) / 255.0
    mask = (img[:, :, 0] < 0.5) & (img[:, :, 1] > 0.5) & (img[:, :, 2] < 0.5)
    non_black = (img[:, :, 0] > 0.0).sum()
    non_black, mask.sum()
    ratio = mask.sum() / non_black
    return ratio > 0.3
