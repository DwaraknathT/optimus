# -------------------------------------------------- #
# This is a Docker image dedicated to develop optimus.
# -------------------------------------------------- #

ARG DOCKER_VERSION=22.09
# Use a base image with cuda 12.0
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

# Basic updates and install build essentials and python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bc git-lfs build-essential gcc wget libssl-dev software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install poetry

# Setup python env. Cache the deps and only rebuild if lock or toml 
# files are changed. 
WORKDIR /deps
COPY requirements.txt /deps/
RUN pip3 install -r requirements.txt

# Install cmake  
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"

# Install nsight compute profiler
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends nsight-compute-2022.4.1 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/nvidia/nsight-compute/2022.4.1:${PATH}"

WORKDIR /workspace/host/optimus