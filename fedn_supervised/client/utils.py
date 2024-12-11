import os
import random

from hydra import initialize, compose

def get_dataset_indices(ds_indices: int) -> list[int]:
    """
    Slurm sets env variables NUM_AGENTS and AGENT_ID to distribute the dataset among agents.
    This function retrieves the indices for the dataset for the current agent.
    All indices sets are disjoint and randomized.
    """
    num_agents = int(os.environ["NUM_AGENTS"])
    agent_id = os.environ["AGENT_ID"]
    agent_id = int(agent_id)

    random.shuffle(ds_indices)
    start_idx = agent_id * int(len(ds_indices) / int(num_agents))
    end_idx = start_idx + int(len(ds_indices) / int(num_agents))
    print(start_idx, end_idx, len(ds_indices))
    return ds_indices[start_idx:end_idx]

def get_hydra_conf():
    """
    Workaround for Hydra initialization in the training script while preserving the first arguments for fedN.
    """
    with open("overrides.txt", "r") as f:
        overrides = f.read().split("\n")
    if overrides[-1] == "":
        overrides = overrides[:-1]
    
    with initialize(config_path="config", job_name="CLIP", version_base="1.1"):
        cfg = compose(config_name="CLIP", overrides=overrides)
    return cfg


def preprocess_image(image, box):
    """
    Preprocesses the given image by cropping it to the specified bounding box and converting it to RGB format.

    Args:
        image (PIL.Image.Image): The input image to be preprocessed.
        box (tuple): A tuple of four integers (left, upper, right, lower) defining the bounding box to crop the image.

    Returns:
        PIL.Image.Image: The preprocessed image in RGB format.
    """
    image = image.crop((box[0], box[1], box[2], box[3]))
    image = image.convert("RGB")
    return image