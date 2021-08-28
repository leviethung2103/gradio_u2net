# Image Segmentattion
* Update: Aug 28, 2021 

## Changelog 
* Aug-28: Change the Docker base image to reduce the size of application, changed to `nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04`, downgrade the version of Pytorch due to CUDNN problem. Reduced from 12 GB -> 3.67 GB

Source Code is saved on server at `/mnt/gradio_u2net` 

## Development 
Sử dụng môi trường dev. Yêu cầu máy tính đã cài đặt sẵn Anaconda để dễ quản lý môi trường và thư viện. Cách cài đặt Anaconda khá đơn giản, nên mình không viết trong này. Có thể tham khảo qua Youtube, Google. 

### 1. Tạo môi trường mới 
```bash
# create a new environment
conda install -n pytorch_p36_gradio cudnn=7.6.5
# activate environment 
conda activate pytorch_p36_gradio
# install package
pip install -r requirements2.txt 
pip install paddlepaddle==2.1.2
pip install paddlehub==2.1.0
pip install gradio==2.2.8
```

### 2. Các thư viện 
Sau đây là các thư viện quan trọng cần chú ý 
* torch==1.6.0
* torchvision==0.7.0
* opencv-pyhton=4.2.0.34


## Build The Docker Image
Build thành docker image phục vụ cho môi trường production/deploy .

### 1. Build Docker Image For API Service - REST API 
Bản build này được xây dựng dựa trên base image: `nvcr.io/nvidia/pytorch:21.05-py3`. Chương trình sử dụng file `test_api.py` làm entry point. 
REST API được xây dựng trên Fast API Framework. Muốn phát triển thêm, chỉnh sửa source code thì vào  file `test_api.py` để phát triển thêm.
Lưu ý, port mặc định của ứng dụng khi build là port 5000. Nếu muốn thay đổi port khác thì vào chỉnh sửa `gpu/Dockerfile_yolov5`. 

**Build Docker Image** 
```bash
# 3.67 GB
docker build -f gpu/Dockerfile_nvidiaruntime -t gradio_u2net:0.0.6 .
```

**Run the application**
Nhớ sử dụng thêm flag `--gpus all` để sử dụng gpu. 
```bash
docker run -itd --restart always -p 5000:5000 --gpus all gradio_u2net:0.0.6
```

### 2.  Build Docker Image for Gradio Website - Demo Only 
Có thêm dịch vụ build Gradio App để test thử nghiệm sản phẩm.
Note: Hiện tại model bên Gradio chỉ support CPU mode. 

** Build the Docker Image** 
```bash
docker build -f cpu/Dockerfile -t gradio_u2net_web:0.0.1 .
```

**Run the application**
```bash
docker run -itd --restart always -p 5002:5002 gradio_u2net_web:0.0.1

```

Bugs: If you've encounterd any bugs related to MKL Library, you can try another build choices

Build with CPU | Model is using at CPU mode

```bash
docker build -f cpu/Dockerfile -t gradio_u2net:0.0.1 .
```

Build with GPU | Model is using at GPU mode
```bash
docker build -f gpu/Dockerfile -t gradio_u2net:0.0.1 .
docker build -f cpu/Dockerfile2 -t gradio_u2net:0.0.1 .
```