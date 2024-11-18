# PlantMap
PlantMap: federated learning for segmentation, classification, and detection of weed species in aerial images taken from farm fields for WASP course 2024.


## Setup Environment

```
conda create --prefix <path to new conda env> python=3.12
conda activate <path or name>
conda install -c conda-forge huggingface_hub
cd /tmp # or any empty directory
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install torch torchvision numpy
pip install -e .
pip install -e ".[notebooks]"
pip install scipy

```

Place all images in data/Hyperlapse


## Create Masks

```
mkdir masks
mkdir test_results
python huggingface_sam_test.py 

```

## Create Flower Dataset for Unsupervised Classification

```
conda activate <path or name>
python create_all_flowers.py
```

