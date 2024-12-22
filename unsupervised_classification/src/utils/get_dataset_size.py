import torch
import os


def get_dataset_size(files: list[str]) -> int:
    file_names = os.listdir('data')
    counter = 0
    for file_name in file_names:
        path = os.path.join('data', file_name)
        images = torch.load(path)
        counter += len(images)
    return counter