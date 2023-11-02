# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
#FROM nvidia/cudagl:11.4.2-base-ubuntu20.04
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer=kwaki

RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install git
#RUN apt-get -y install libglfw3 libglew2.0 freeglut3-dev
# #RUN apt-get -y install libglfw3 libglew2.1 freeglut3-dev
#RUN apt-get -y install libglvnd0 libgl1 libglx0 libegl1
#RUN apt-get -y install libgl1
#RUN apt-get -y install libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev
#RUN apt-get -y install libxext6 libx11-6
# RUN apt-get -y install cmake
# RUN apt-get -y install blender
# RUN apt-get -y install subversion
RUN apt-get -y install ffmpeg libgl1-mesa-glx

# # Env vars for the nvidia-container-runtime.
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
# 
RUN pip install -U numpy
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install scipy
#RUN pip install sklearn
# RUN pip install pytorch3d
# RUN pip install future-fstrings
# RUN pip install numpy-stl
# RUN pip install pyassimp
# RUN pip install absl-py
# RUN pip install sklearn
# RUN pip install pygltflib
# RUN pip install gym
# RUN apt-get install -y glmark2
# # RUN pip install bpy
# 
# 
# RUN apt-get update \
#   && apt-get install -y -qq --no-install-recommends \
#     libxext6 \
#     libx11-6 \
#     libglvnd0 \
#     libgl1 \
#     libglx0 \
#     libegl1 \
#     freeglut3-dev \
#   && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
# # Add gym.
# RUN pip install --upgrade pip \
#   && pip install gym[box2d]
# 
# # WORKDIR /blender
# # cmake -DWITH_PYTHON_INSTALL=OFF -DWITH_AUDASPACE=OFF -DWITH_PYTHON_MODULE=ON -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/python3.7/ -DWITH_X11_XINPUT=OFF -DWITH_FFTW3=ON ..
# 
# # apt-get install zstd libzstd-dev
# # libfreetype6-dev
# # libopencolorio-dev
# # had to checkout an older version v2.83.10. latest blender uses opencolorio v2
# # apt-get install libxrender-dev
# # libboost-all-dev
# 
# WORKDIR /avian3d
# 
# # RUN git clone git@github.com:marcbadger/avian-mesh.git
# RUN git clone https://github.com/marcbadger/avian-mesh.git
# WORKDIR /avian3d/avian-mesh
# RUN pip install -r requirements.txt
# 
# WORKDIR /workspace
