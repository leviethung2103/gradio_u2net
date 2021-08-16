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

Remove: scikit-image==0.14.0, scipy==1.1.0

```bash
docker run -it -p 5002:5002 gradio_u2net:0.0.1
# docker run -it -p 5002:5002 --gpus all gradio_u2net:0.0.1
```