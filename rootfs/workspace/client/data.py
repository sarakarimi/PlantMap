from os import environ
from pathlib import Path

from math import floor

from torch import load, save
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

root_path = Path(__file__).absolute().parent

#### !!!!!!!!!!!!ATTENTION!!!!!!!!!!!! ####
####
####        This is a very rough draft
####    of how we could split/fetch data
####     it's called once when running
####         on initial client run
#### 
#### !!!!!!!!!!!!ATTENTION!!!!!!!!!!!! ####

def get_data(out_dir=Path("data")):
    # Make dir if necessary
    out_dir.mkdir(exist_ok=True)

    # Only download if not already downloaded
    tmp = out_dir/"train"
    if not tmp.exists():
        MNIST(root=tmp, transform=ToTensor, train=True, download=True)
    tmp = out_dir/"test"
    if not tmp.exists():
        MNIST(root=tmp, transform=ToTensor, train=False, download=True)


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = Path(environ.get("FEDN_DATA_PATH", root_path/"data/clients/1/mnist.pt"))

    data = load(data_path, weights_only=True)

    return data


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def split(out_dir="data"):
    n_splits = int(environ.get("FEDN_NUM_DATA_SPLITS", 2))

    # Make dir
    Path(f"{out_dir}/clients").mkdir(exist_ok=True)

    # Load and convert to dict
    train_data = MNIST(root=f"{out_dir}/train", transform=ToTensor, train=True)
    test_data = MNIST(root=f"{out_dir}/test", transform=ToTensor, train=False)
    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(train_data.targets, n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(test_data.targets, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = Path(f"{out_dir}/clients/{str(i+1)}")
        subdir.mkdir(exist_ok=True)
        save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            subdir/"mnist.pt",
        )


if __name__ == "__main__":
    # Prepare data if not already done
    if not (root_path/"data/clients/1").exists():
        get_data()
        split()
