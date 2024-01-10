FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /segmentation/src/wav2vec2_multiclass/

ENV SHELL=/bin/bash

ADD ./requirements.txt ./
ADD ./wav2vec2_multiclass.py ./
ADD ./spotify-lab.Dockerfile ./

COPY . .

RUN pip install -r requirements.txt 
