FROM debian:bullseye-slim

LABEL maintainer="Anaconda, Inc"
# First stage: Build dependencies using Ubuntu 20.04
# FROM ubuntu:20.04 AS builder
# FROM python:3.9-alpine
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
FROM --platform=linux/amd64 ubuntu:20.04
ENV TZ=Asia/Kolkata
ENV DEBIAN_FRONTEND=noninteractive
# FROM --platform=linux/amd64  ubuntu:20.04 python:3.9-alpine
# export DOCKER_DEFAULT_PLATFORM=linux/amd64
# hadolint ignore=DL3008
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        openssh-client \
        procps \
        subversion \
        libarchive-dev \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh -q && \
    echo "4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e miniconda.sh" > shasum && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy


WORKDIR /app

COPY ./ilastikApi/requirements.txt /ilastik_requirements.txt

# RUN /opt/conda/bin/conda create -n myenv python=3.9 && \
#     echo "conda activate myenv" >> ~/.bashrc
# conda install -c conda-forge mamba
RUN conda config --add channels conda-forge && \
    conda update -n base --all && \
    conda install -n base mamba  && \
    mamba install --override-channels -c pytorch -c ilastik-forge -c conda-forge ilastik

# additional dependencies
# netcat used for checking for resources on the network before start, see start.sh for use
RUN apt-get update && \
    apt-get install -y netcat

# Install dependencies to solve dependency bug: openslide
RUN apt install -y openslide-tools && \
    apt install -y build-essential && \
    apt install -y git && \
    apt install -y default-jdk && \
    apt install -y curl && \
    pip install -r /ilastik_requirements.txt && \
    pip install --upgrade pyopenssl

EXPOSE 8001
