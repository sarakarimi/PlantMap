import torch
import numpy as np
import os

from tqdm import tqdm
from PIL import Image


def main() -> None:
    image_folder = "../data/Hyperlapse/"
    mask_folder = "../masks/"

    file_names = os.listdir(image_folder)

    flowers = []
    with torch.no_grad():
        with tqdm(file_names) as pbar:
            for file_name in pbar:
                if not file_name.endswith("JPG"):
                    continue
                file_path = os.path.join(image_folder, file_name)
                mask_file = file_name.replace("JPG", "pt")
                mask_path = os.path.join(mask_folder, mask_file)

                image_array = np.array(Image.open(file_path))
                if not os.path.exists(mask_path):
                    continue
                masks = torch.load(mask_path)

                for i, mask in enumerate(masks):
                    bbox = mask["bbox"]
                    segmentation = mask["segmentation"].toarray()
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
                    cropped_image = image_array[
                        y_min : y_min + height, x_min : x_min + width
                    ]
                    flowers.append(cropped_image)

                    pbar.set_description_str(f"Progress: {i} / {len(masks)}")
                torch.save(flowers, f"data2/{file_name}.pt")
                flowers = []


if __name__ == "__main__":
    main()
