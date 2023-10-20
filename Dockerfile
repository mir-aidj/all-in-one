FROM nvidia/cuda:12.2.2-base-ubuntu22.04

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git build-essential ffmpeg

RUN pip install ninja allin1 git+https://github.com/CPJKU/madmom
RUN pip install install natten -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app