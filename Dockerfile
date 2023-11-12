FROM python:3.10-slim as system

# base image
RUN useradd --create-home --shell /bin/bash allin1_user
RUN apt update && apt upgrade -y
RUN apt install build-essential gcc git ffmpeg -y
RUN pip3 install --upgrade pip setuptools

# install heavy packages
FROM system as application

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install natten -f https://shi-labs.com/natten/wheels/cpu/torch2.0.0/index.html 
RUN pip3 install git+https://github.com/CPJKU/madmom


FROM application as runtime
WORKDIR /home/allin1_user
COPY . .
RUN pip3 install demucs hydra-core omegaconf huggingface_hub matplotlib pytest
# NOTE install additional packages here
