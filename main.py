# Use the Paddle Hub to download the model
import cv2
import paddlehub as hub
import gradio as gr
import time 
from decouple import config

# Load / Download model from Paddle Hub
model = hub.Module(name="U2Net")

SERVER_HOST = config("SERVER_HOST")
SERVER_PORT = config("SERVER_PORT",cast=int)
PUBLIC_SHARE= config("PUBLIC_SHARE", cast=bool)

def infer(img):
    _start_time = time.time()
    result = model.Segmentation(
        images=[cv2.imread(img.name)], paths=None, batch_size=1, input_size=320, output_dir="output", visualization=True
    )
    print ("Processing time", time.time() - _start_time)
    return result[0]["front"][:, :, ::-1], result[0]["mask"]


inputs = gr.inputs.Image(type="file", label="Original Image")
outputs = [gr.outputs.Image(type="numpy", label="Front"), gr.outputs.Image(type="numpy", label="Mask")]

title = "Image Segmentation"
gr.Interface(infer, inputs, outputs, title=title, theme="compact", server_name=SERVER_HOST, server_port=SERVER_PORT).launch(share=PUBLIC_SHARE)
