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

### Federated Learning and FedN

Federated learning (FL) is a decentralized approach to machine learning that allows multiple parties to collaboratively train a shared model without sharing their raw data. This technique is particularly valuable in scenarios where data privacy, security, or locality is critical, such as healthcare, finance, and agriculture.

#### **How Federated Learning Works**
1. **Local Training**: Each participating client (e.g., farmers, edge devices) trains a local model using its private data. This training process occurs independently and securely on each client’s device.
2. **Model Updates**: Once training is complete, each client sends only the model updates (e.g., weights, gradients) to a central server. Importantly, raw data remains on the client’s device, ensuring privacy.
3. **Global Aggregation**: The central server aggregates the updates from all participating clients to create a global model. This global model is then shared back with the clients for the next round of training.
4. **Iterative Process**: Steps 1–3 are repeated for several rounds until the global model converges to a satisfactory performance level.

Federated learning comes with advantages such as **(i) privacy**: as sensitive data never leaves the client’s device, reducing the risk of data breaches, **(ii) scalability**: as it can scale to millions of devices or participants, enabling collaborative training on massive datasets, and **(iii) personalization**: as clients can fine-tune the global model locally to adapt it to their specific data distribution.

#### **Aggregation Methods in Federated Learning**
A critical aspect of FL is the aggregation of model updates to ensure the global model improves with each round. Common aggregation methods include:
- **Federated Averaging (FedAvg)**: A simple yet effective method introduced by Sun et al. [1], where the global model is updated by averaging the weights or gradients received from clients, weighted by the size of each client's dataset.
- **Gradient Aggregation**: In scenarios where gradients are shared, these can be aggregated directly to update the global model.
- **Adaptive Aggregation**: Advanced methods that account for heterogeneity in client data, ensuring that clients with diverse data distributions contribute effectively to the global model.

#### **FedN for Federated Weed Detection**
In this project, we leverage [**FedN**](https://www.scaleoutsystems.com/framework), a robust federated learning framework designed for scalable and efficient model training. Using FedN, we aggregate locally trained models from farmers to create a unified global model for weed detection and classification. Each farmer's model is trained on their specific annotated data (e.g., clovers or chamomiles), and the aggregated model benefits from the diverse local datasets while respecting data privacy.
This approach allows us to build a high-performing foundation model for farm weed detection without compromising individual data security.


## Machine Learning Methods
Since the solution entails performing **image segmentation and object detection/classification** tasks on images, the following sections detail the machine learning approaches employed for each specific task.

### Segmentation of images
We use the  to perform segmentation on the images. 
To perform image segmentation, we leverage the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), a versatile and state-of-the-art tool designed for robust segmentation across diverse image datasets.


### Classification of objects in the images
With the segmentations in place, we explore two approaches for the classification task:

1. **Unsupervised Classification** : This method relies on feature matching to group segments of the original image that match with a given example without requiring labeled data.

2. **Supervised Classification** : This approach utilizes the pre-trained [CLIP](https://github.com/openai/CLIP) model, leveraging its powerful multi-modal capabilities to classify segments based on learned visual and the provided textual prompts.

#### (Semi-/Un-)supervised Classification
TODO add description of the method

We use SAM in order to get flower images without background as this is usually pretty accurate using a pretrained SAM2 model. 

For every possible wildflower, we select one candidate as reference image. 
We finetune a masked-autoencoder, discard the decoder and use the encoder part of the model to retrieve image features. 

To train the model, we chose to compare two different approaches, [BYOL](https://arxiv.org/abs/2006.07733) and [SimCLR](https://arxiv.org/abs/2002.05709), both methods for learning visual representations. 
An encoder model maps each image into a vector. By normalizing the vector (unit vector) and comparing them using cosine-similarity, we get a probability of both images belonging to the same class or not. 

##### Challenges

* The number of distinct flower classes is comparably low compared to the number of overall images
* Just by looking on the raw data, it is clear that the dataset is not evenly distributed, i.e., unbalanced. 
* Many flowers of different classes look similar, e.g., all flowers with white blossoms. 
* SAM is not perfect, i.e., it predicts false positives and 



#### Supervised Classification
TODO add description of the method

## Datasets
Since we started with an unannotated dataset of raw images, it was necessary to first annotate and label the dataset to generate training data required for training an image object detection and classification model. To achieve this, we used the pre-trained SAM+CLIP approach described in the **Method** section.
Specifically, the raw images were fed into the pre-trained **SAM** model, which performed segmentation to identify distinct regions in the images. 
These segmented regions were then passed to the **CLIP** model for classification. The CLIP model was provided with a list of textual prompts representing the labels of existing weed species, enabling it to classify each segment according to the specified labels. 
The resulting labeled dataset has over 2000 samples with labels for four weed species **Daisy, Yarrow, Dandelion, and Red clover**. The dataset is published in the HuggingFace dataset repository [PlantMap](https://huggingface.co/datasets/sarakarimi30/PlantMap).

To develop a base model better suited for detection and classification on the above-mentioned dataset, we pre-trained our model on a similar dataset, the [**Eindhoven Wildflower Dataset**](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/U4VQJ6). This dataset contains 2,002 high-resolution annotated images of wildflowers, providing a robust starting point for training.





## Experiments & Results
TODO

## Conclusions & Future Work
TODO

## References
[1] Sun, Tao, Dongsheng Li, and Bao Wang. "Decentralized federated averaging." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.4 (2022): 4289-4301.