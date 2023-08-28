FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    python3 \
    python3-pip \
    git

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /home/