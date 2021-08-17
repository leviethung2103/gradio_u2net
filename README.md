# Gradio App - U2-Net
Build with CPU | Model is using at CPU mode
```bash
docker build -f cpu/Dockerfile -t gradio_u2net:0.0.1 .
```
Build with GPU | Model is using at GPU mode
```bash
docker build -f gpu/Dockerfile -t gradio_u2net:0.0.1 .
docker build -f cpu/Dockerfile2 -t gradio_u2net:0.0.1 .
```

# Installation
Need to install the cudatoolkit 10.1, Cudnn 7.6.5 to use the paddle-gpu model
```bash
conda install -n pytorch_p36_gradio cudnn=7.6.5
conda install -n pytorch_p36_gradio -c anaconda cudnn=7.6.5
```

Remove: scikit-image==0.14.0, scipy==1.1.0

```bash
docker run -it -p 5002:5002 gradio_u2net:0.0.1
docker run -it -p 5000:5000 --gpus all gradio_u2net:0.0.2
```

```bash
docker run -itd --restart always -p 5000:5000 --gpus all gradio_u2net:0.0.2
docker build -f gpu/Dockerfile_yolov5 -t gradio_u2net:0.0.2 .
docker run -it -p 5000:5000 --gpus all gradio_u2net:0.0.2
```