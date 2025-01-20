import torch
import os


def get_dataset_size(files: list[str]) -> int:
    """
    Helper function to get the size of the dataset.
    """
    file_names = os.listdir('../data2')
    counter = 0
    for file_name in file_names:
        path = os.path.join('../data2', file_name)
        images = torch.load(path)
        counter += len(images)
    return counter