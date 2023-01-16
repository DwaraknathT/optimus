# -------------------------------------------------- #
# This is a Docker image dedicated to develop optimus.
# -------------------------------------------------- #

ARG DOCKER_VERSION=22.09
# Use a base image with cuda 12.0
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu20.04
FROM ${BASE_IMAGE}

# Basic updates and upgrades.
RUN apt-get update && \
    apt-get install -y --no-install-recommends bc git-lfs&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# backend build
WORKDIR /workspace/optimus
ADD . /workspace/optimus


