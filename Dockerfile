FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer=kwaki

RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install git

RUN apt-get -y install ffmpeg libgl1-mesa-glx

RUN pip install -U numpy
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install scipy

WORKDIR /workspace
