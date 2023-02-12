# -------------------------------------------------- #
# This is a Docker image dedicated to develop optimus.
# -------------------------------------------------- #

ARG DOCKER_VERSION=22.09
# Use a base image with cuda 12.0
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

# Basic updates and install build essentials and python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bc git-lfs build-essential gcc wget libssl-dev software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install poetry 

# Install cmake  
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"

WORKDIR /workspace/host/optimus