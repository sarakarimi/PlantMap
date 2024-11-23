#!/opt/conda/bin/python3


from math import ceil
from pathlib import Path

from sys import path

import torch
from model import load_parameters, save_parameters

from data import load_data
from fedn.utils.helpers.helpers import save_metadata


root_path = Path(__file__).absolute().parent
path.append(root_path)

def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    data = load_data(data_path, batch_size)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    model.train(data=data_path, #path to yaml file  
           imgsz=data["size"], #image size for training  
           batch=batch_size, #number of batch size  
           epochs=epochs, #number of epochs  
           device=0,
           lr0=lr) #device ‘0’ if gpu else ‘cpu’

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": data["nbatch"],
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
