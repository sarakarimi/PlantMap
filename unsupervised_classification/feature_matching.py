import torch
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn

from PIL import Image
from transformers import ViTMAEModel, ViTImageProcessor


def extract_features(processor: nn.Module, model: nn.Module, images: list[torch.Tensor]) -> torch.Tensor:
    """
    Retrieve the last hidden state of the model and pool the features.
    """
    inputs = processor(images=images, return_tensors="pt")
    inputs["pixel_values"].shape
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).squeeze()  # Pool features
    features = features
    return features


def main() -> None:
    """
    Naive approach to test the similarity between images using the ViTMAE model using a pretrained model. 
    """
    model_name = "facebook/vit-mae-base"  # Pretrained ViTMAE model
    model = ViTMAEModel.from_pretrained(model_name)
    model.eval()

    processor = ViTImageProcessor.from_pretrained(model_name)

    image_path = "data/HYPERLAPSE_0001.JPG.pt"

    images = torch.load(image_path)

    features = extract_features(processor, model, images)
    features_transposed = features.transpose(0, 1)
    similarities = features @ features_transposed
    similarities = similarities.sigmoid().numpy()

    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]):
            if similarities[i, j] < 0.5:
                subplots = plt.subplots(1, 2)
                subplots[1][0].imshow(images[i])
                subplots[1][1].imshow(images[j])
                plt.savefig(f"similarities/{i}_{j}.png")

    return similarities


if __name__ == "__main__":
    main()