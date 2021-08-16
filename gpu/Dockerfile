FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx libsm6 libxext6 libxrender1

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install -r requirements.txt 
RUN pip install paddlepaddle-gpu==2.1.2

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app
# Expost the port
EXPOSE 5002

CMD ["python3", "main.py"]
