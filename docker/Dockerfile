
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=user \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:iscilab" | chpasswd

## basic tools
RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    cmake \
    wget \
    make \
    vim \
    build-essential \
    libblkid-dev \
    libssl-dev \
    e2fslibs-dev \
    libboost-all-dev \
    libaudit-dev \
    tzdata \
    libgl1-mesa-glx \
    libaio-dev \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN pip3 install --upgrade pip

## Dependency installation
RUN pip3 install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==24.0
RUN pip3 install flash-attn --no-build-isolation

COPY ./setup.sh /
RUN sh /setup.sh
RUN rm /setup.sh

COPY requirements.txt /
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

## setting
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL) ALL" > /etc/sudoers
RUN echo "root ALL=(ALL) ALL" > /etc/sudoers

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}
RUN ls
