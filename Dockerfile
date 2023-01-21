ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime


RUN apt-get update && apt-get install -y wget build-essential unzip zlib1g-dev ffmpeg libsm6 libxext6 libjpeg-dev git \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN bash -c "pip install --upgrade pip \
    && pip install -r /requirements.txt \
    && rm /requirements.txt"

WORKDIR /opt
