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

RUN git clone https://github.com/facebookresearch/sam2.git
RUN git clone https://github.com/scaleoutsystems/fedn.git

RUN mkdir /workspace/sam2/sam2/{masks,test_results} similarities
RUN mv /workspace/huggingface_sam_test.py /workspace/sam2/sam2/

# huggingface hub
RUN pip install huggingface_hub scipy ultralytics fedn

# FedN libraries
WORKDIR /workspace/fedn
RUN pip install -e .

# SAM2
# https://github.com/facebookresearch/sam2
WORKDIR /workspace/sam2
RUN pip install -e .
RUN pip install -e ".[notebooks]"
#RUN cd checkpoints 
#RUN ./download_ckpts.sh
# move everything in checkpoints to correct path
#RUN mv /workspace/sam2/checkpoints/* /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/

WORKDIR /
