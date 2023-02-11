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
    apt-get install -y python3.11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cmake.
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz && \
    tar -zxvf cmake-3.20.0.tar.gz && \
    cd cmake-3.20.0 && ./bootstrap && make && make install

WORKDIR /workspace/optimus
ADD . /workspace/optimus
