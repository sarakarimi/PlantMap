import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import preprocess_image

class ImageTextDataset(Dataset):
    """
    A custom dataset class for handling image and text data for CLIP model training.

    Args:
        images (list): A list of preprocessed images.
        text_labels (list): A list of text labels corresponding to the images.
        categories (list): A list of unique categories for the labels.
        processor (CLIPProcessor): A processor for preparing the data for the CLIP model.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Returns a single sample from the dataset at the given index.
    """

    def __init__(self, images, text_labels, categories, processor):
        self.images = images
        self.text_labels = text_labels
        self.processor = processor
        self.categories = categories

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text_label = self.text_labels[idx]
        label = self.categories.index(text_label)
        processed = self.processor(
            text=text_label, images=image, return_tensors="pt", padding=True
        )

        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": label,
        }


def collate_fn(batch):
    """
    Collates a batch of data for a model that processes both images and text.

    Args:
        batch (list of dict): A list where each element is a dictionary containing:
            - "pixel_values" (torch.Tensor): The tensor representing image pixel values.
            - "input_ids" (torch.Tensor): The tensor of input token IDs for text.
            - "attention_mask" (torch.Tensor): The tensor of attention masks for text.
            - "labels" (int or torch.Tensor): The label associated with the data.

    Returns:
        dict: A dictionary with the following keys:
            - "pixel_values" (torch.Tensor): A stacked tensor of image pixel values.
            - "input_ids" (torch.Tensor): A padded tensor of input token IDs.
            - "attention_mask" (torch.Tensor): A padded tensor of attention masks.
            - "labels" (torch.Tensor): A tensor of labels.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,  # Padding token ID
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.tensor([item["labels"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# TODO: annotate correct return types
def preprocess_dataset(dataset, indices: list[int]) -> tuple[list, list, list]:
    """
    Preprocesses a dataset of images by cropping them based on bounding boxes and extracting labels.

    Args:
        dataset (list of dict): A list of dictionaries where each dictionary represents an image file.
            Each dictionary should contain the following keys:
            - "image": The image data.
            - "labels": A list of labels corresponding to objects in the image.
            - "boxes" or "bboxes": A list of bounding boxes for the objects in the image.

    Returns:
        tuple: A tuple containing:
            - cropped_images (list): A list of cropped images based on the bounding boxes.
            - labels (list): A list of labels corresponding to the cropped images.
            - categories (set): A list of unique categories found in the dataset.
    """
    cropped_images = []
    labels = []
    categories = set()
    box_str = "boxes" if "boxes" in dataset[0] else "bboxes"

    with tqdm(indices, desc="Preprocessing dataset") as pbar:
        for index in pbar:
            file = dataset[index]
            for label, box in zip(file["labels"], file[box_str]):
                cropped_images.append(preprocess_image(file["image"], box))
                categories.add(label)
                labels.append(label)

    return cropped_images, labels, list(categories)