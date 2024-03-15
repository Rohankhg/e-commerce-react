FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /app

COPY ./mainApi/requirements.txt /requirements.txt
COPY ./mainApi/opencv_lib/* /opencv_lib/*

# additional dependencies
# netcat used for checking for resources on the network before start, see start.sh for use
RUN apt-get update && \
    apt-get install -y netcat-traditional

# Install dependencies to solve dependency bug: openslide
RUN apt install -y openslide-tools && \
    apt install -y build-essential && \
    apt install -y git && \
    apt install -y default-jdk && \
    pip install numpy==1.23.5 && \
    pip install -r /requirements.txt

RUN apt-get update && \
    apt-get install -y libopencv-dev

COPY ./mainApi/opencv_lib/* /opencvlib/

RUN apt update && apt install -y cmake g++ wget unzip
RUN cp -r /opencvlib/* /usr/local/lib/
RUN ldconfig /usr/local/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

EXPOSE 8000
