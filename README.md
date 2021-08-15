# Gradio App - U2-Net

```bash
docker build -f cpu/Dockerfile -t gradio_u2net:0.0.1 .
```


```bash
docker run -it --ipc=host --gpus all gradio_u2net:0.0.1
```