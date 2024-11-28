
## Dataset annotation using SAM+CLIP
As we have an unannotated dataset, we use this SAM segmentation + CLIP classification to annotate our dataset.
The original un-annotated dataset is stored in [this](https://huggingface.co/datasets/gotdairyya/plant_images) HuggingFace repository. By running `python sam_clip/sam_clip_text_seg.py` scripts we created an annotated dataset save in [this](https://huggingface.co/datasets/sarakarimi30/PlantMap) HuggingFace repository.

Before running the scripts make sure you download SAM and CLIP checkpoints using this script`sam_clip/download_pretrained_ckpt.sh`

## Credits
Parts of the code in `sam_clip` directory is taken from [this](https://github.com/MaybeShewill-CV/segment-anything-u-specify) repository.
