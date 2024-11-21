FROM pytorch/pytorch

COPY rootfs /

WORKDIR /workspace

# Installing required and useful packages from repositories
# https://docs.humansignal.com/tutorials/segment_anything_2_image
RUN apt update && apt install -y \
    gcc \
    git \
    openjdk-8-jdk \
    vim \
    curl \
    wget \
    iproute2 \
    iputils-ping \
    unzip

RUN git clone https://github.com/HumanSignal/label-studio-ml-backend.git
RUN git clone https://github.com/facebookresearch/sam2.git

RUN mkdir masks test_results similarities

# huggingface hub
RUN pip install huggingface_hub scipy

# Label studio
WORKDIR /workspace/label-studio-ml-backend
RUN pip install -e .
RUN pip install -r ./label_studio_ml/examples/segment_anything_2_image/requirements.txt

# SAM2
# https://github.com/facebookresearch/sam2
WORKDIR /workspace/sam2
RUN pip install -e .
RUN pip install -e ".[notebooks]"
#RUN cd checkpoints 
#RUN ./download_ckpts.sh
# move everything in checkpoints to correct path
#RUN mv /workspace/sam2/checkpoints/* /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/

RUN mkdir -p /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/segment-anything-2/checkpoints
WORKDIR /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/segment-anything-2/checkpoints
RUN wget "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
# go back to examples
WORKDIR /workspace/label-studio-ml-backend/label_studio_ml/examples

# maybe we run this
RUN label-studio-ml start ./segment_anything_2_image

WORKDIR /
