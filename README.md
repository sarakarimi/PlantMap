# PlantMap: Federated learning for segmentation, classification, and detection of weed species in aerial images taken from farm fields



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

#### Create Masks

```
# Place all images in data/Hyperlapse
mkdir masks
mkdir test_results
python scripts/huggingface_sam_test.py 

```

#### Create Flower Dataset for Unsupervised Classification

```
mkdir masks
conda activate <path or name>
python unsupervised_classification/create_all_flowers.py
```


#### Similarity Matching
Due to laziness, the current code does only compare all flowers of a single picture. 

```
# Run the "Create Flower Dataset for Unsupervised Classification" step
mkdir similarities

conda activate <path or name>
python unsupervised_classification/feature_matching.py
```

------------------------------------------------------------------------------------------------------------------------------------------
### Project members:

Derya Akbaba - Linkoping University <br>
Sofia Andersson - Lund University <br>
Sara karimi - KTH Royal Institute of Technology <br>
Markus Fritzsche - Linkoping University <br>
Xavante Erickson -  <br>

## Introduction
TODO

## Methods
TODO


### Segmentation of objects in the images
We use the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to perform segmentation on the images. 

### Classification of flower species in the images
With segmentations at hand, we explore two approaches for the classification part: <br>
(1) unsupervised classification method that uses feature matching, and (2) supervised classification that uses pre-trained [CLIP](https://github.com/openai/CLIP) model.


#### 1. Unsupervised Classification
TODO add description of the method


#### 2. Supervised Classification
TODO add description of the method

## Data Annotation


## Experiments & Results
TODO

## Conclusions & future work
TODO

## References