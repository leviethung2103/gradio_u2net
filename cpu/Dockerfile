FROM nvcr.io/nvidia/pytorch:21.05-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx libsm6 libxext6 libxrender1

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt 
RUN pip install --no-cache paddlepaddle==2.1.2

# # install mkl-dnn -> cannot run the app with mkl 
# # Intel MKL function load error: cpu specific dynamic library is not loaded.
# WORKDIR /root
# # https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Asia/Ho_Chi_Minh
# RUN apt-get install -y cmake doxygen wget
# RUN git clone https://github.com/01org/mkl-dnn.git
# WORKDIR mkl-dnn
# # RUN cd scripts && ./prepare_mkl.sh && cd ..
# RUN mkdir -p build && cd build && cmake .. && make -j4
# WORKDIR build
# RUN make install

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

RUN cp -r .env.example .env 

# Set environment variables
ENV HOME=/usr/src/app
# Expost the port
EXPOSE 5002

CMD ["python3", "main.py"]
