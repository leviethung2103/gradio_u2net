# Author: Hung Le Viet 
# This file is used to server the U2_Net Segmentation Model 
# Utillize the FastAPI to host the application
# The model is originally served from the Paddle Hub Service

import sys
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import cv2
from starlette.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import numpy as np
import io
import base64
import cv2
import paddlehub as hub
import gradio as gr
import torch
import time 
from decouple import config
from datetime import datetime

# Load the model 
seg_model = hub.Module(name="U2Net")
app = FastAPI()


class ImageSegmentation(BaseModel):
    image: str

@app.post("/api/image-segmentation/from-file")
async def image_segmentation_from_file(file: UploadFile = File(...)):
    _start_time = time.time()
    fimage = await file.read()
    image = np.frombuffer(fimage, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # -> image is BGR 

    # inference
    result = seg_model.Segmentation(
        images=[image], paths=None, batch_size=1, input_size=320, output_dir="output", visualization=True
    )
    # foreground = result[0]["front"][:, :, ::-1]
    foreground = result[0]["front"]
    mask = result[0]["mask"]

    res, im_png = cv2.imencode(".png", foreground)

    print (f"{datetime.now()} - Processing Time: {time.time() - _start_time}")

    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.post("/api/image-segmentation/from-base64")
async def image_segmentation_from_json(input: ImageSegmentation):
    p2s_request = jsonable_encoder(input)
    image = base64.b64decode(p2s_request["image"])
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_ANYCOLOR)

    # inference
    result = seg_model.Segmentation(
        images=[image], paths=None, batch_size=1, input_size=320, output_dir="output", visualization=True
    )
    foreground = result[0]["front"][:, :, ::-1]
    mask = result[0]["mask"]

    res, im_png = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    API_HOST = config("API_HOST", default="0.0.0.0")
    API_PORT = config("API_PORT",default=5000,cast=int)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
