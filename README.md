# PlantMap: Federated learning for segmentation, detection, and classification of weed species in aerial images taken from farm fields



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

## Scalable solution
TODO mention we are using FedN and a short intro to federated learning and aggregation methods

## Machine Learning Methods
Since the solution entails performing **image segmentation and object detection/classification** tasks on images, the following sections detail the machine learning approaches employed for each specific task.

### Segmentation of images
We use the  to perform segmentation on the images. 
To perform image segmentation, we leverage the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), a versatile and state-of-the-art tool designed for robust segmentation across diverse image datasets.


### Classification of objects in the images
With the segmentations in place, we explore two approaches for the classification task:

1. **Unsupervised Classification** : This method relies on feature matching to group segments of the original image that match with a given example without requiring labeled data.

2. **Supervised Classification** : This approach utilizes the pre-trained [CLIP](https://github.com/openai/CLIP) model, leveraging its powerful multi-modal capabilities to classify segments based on learned visual and the provided textual prompts.

#### Unsupervised Classification
TODO add description of the method


#### Supervised Classification
TODO add description of the method

## Data Annotation
TODO

## Experiments & Results
TODO

## Conclusions & future work
TODO

## References