import torch
import numpy as np
import os

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset



def main() -> None:
    """
    Helper class to create all flowers from the masks created in a previous step using SAM. 
    """
    mask_folder = "masks/"


    ds = load_dataset("gotdairyya/plant_images")
    images = [ds['train'][i]['image'] for i in range(len(ds["train"]))]

    flowers = []
    with torch.no_grad():
        with tqdm(enumerate(images)) as pbar:
            for idx, image in pbar:
                image_array = np.array(image)
                masks_path = f"{mask_folder}/mask-{idx}.pt"
                if not os.path.exists(masks_path):
                    continue
                masks = torch.load(masks_path)

                for i, mask in enumerate(masks):
                    bbox = mask["bbox"]
                    segmentation = mask["segmentation"]
                    size_x, size_y = segmentation.shape

                    x_min, y_min, width, height = bbox
                    x_min = int(x_min)
                    y_min = int(y_min)
                    width = int(width)
                    height = int(height)

                    masked_image = image_array * segmentation.reshape(size_x, size_y, 1)
                    cropped_image = masked_image[
                        y_min : y_min + height, x_min : x_min + width
                    ]
                    flowers.append(cropped_image)

                    pbar.set_description_str(f"Progress: {i} / {len(masks)}")
                torch.save(flowers, f"data/sample-{idx}.pt")
                flowers = []


if __name__ == "__main__":
    main()
