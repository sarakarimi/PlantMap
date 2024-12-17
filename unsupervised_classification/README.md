# Training

We abandoned these methods due their lack of performance. 


## Pretrain Masked-Autoencoder

This code will pretrain the Masked-Autoencoder model on our dataset to learn relevant feature presentations. 
Unfortunately, in the end pretraining didn't make any significant difference in the model's performance. You can find the default parameters in the files. 

For ```--vit``` you can choose: 

- MAE: for the masked-autoencoder
- DINO: for the dino v2 model
- CLIP: for the CLIP model

```
uv run src/train_auto.py <params>
```

## Learning Similarities.

All ```src/train_sim{1..3}``` models train a model for similarity matching but using different loss functions. 

```
src/train_sim1.py: Use SimCLR loss
src/train_sim2.py: Use BYOL loss
src/train_sim3.py: use MoCO loss 
```