FROM pytorch/pytorch

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

# Label studio
RUN git clone https://github.com/HumanSignal/label-studio-ml-backend.git
RUN cd label-studio-ml-backend
RUN pip install -e .
RUN cd label_studio_ml/examples/segment_anything_2_image
RUN pip install -r requirements.txt

# SAM2
# https://github.com/facebookresearch/sam2
RUN cd /workspace
RUN git clone https://github.com/facebookresearch/sam2.git && cd sam2
RUN pip install -e .
RUN pip install -e ".[notebooks]"
#RUN cd checkpoints 
#RUN ./download_ckpts.sh
# move everything in checkpoints to correct path
#RUN mv /workspace/sam2/checkpoints/* /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/
RUN cd /workspace/sam2/checkpoints/* /workspace/label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image/
RUN mkdir segment-anything-2
RUN cd segment-anything-2
RUN mkdir checkpoints
RUN cd checkpoints
RUN wget "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
# go back to examples
RUN cd /workspace
RUN cd label-studio-ml-backend/label_studio_ml/examples

# maybe we run this
RUN label-studio-ml start ./segment_anything_2_image


