#!/bin/bash
set -x

TAG=metamorph_mujocov3
# PARENT=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
PARENT=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
INSTALL_SCRIPT="install_gpu_deps"
USER_ID=`id -u`

docker build -f docker/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg INSTALL_SCRIPT=${INSTALL_SCRIPT} \
  --build-arg USER_ID=${USER_ID} \
  -t ${TAG} .
