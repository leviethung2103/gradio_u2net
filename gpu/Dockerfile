FROm nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM pure/python:3.7-cuda10.1-cudnn7-runtime
# Install linux packages
# RUN apt update && apt install -y zip htop screen libgl1-mesa-glx libsm6 libxext6 libxrender1

# usd with nvidia/cuda:10.1-cudnn7-runtime-ubuntu
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        python3.6 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libsm6 libxext6 \ 
        libxrender-dev \
        libgomp1 \ 
        && \
    rm -rf /var/lib/apt/lists/* 

# Install python dependencies
COPY requirements2.txt .
RUN python3 -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install -r requirements2.txt 

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app
# Expost the port
EXPOSE 5000

CMD ["python3", "test_api.py"]
