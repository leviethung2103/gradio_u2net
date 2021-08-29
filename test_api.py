# Author: Hung Le Viet
# This file is used to server the U2_Net Segmentation Model
# Utillize the FastAPI to host the application
# I changed the paddlehub model -> native model
# This is an upgrade version from api.py file

# https://blog.csdn.net/wa1tzy/article/details/107015923

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
import os
import torch
import time
from decouple import config
from datetime import datetime
from u2net.u2net_model import U2NET
from u2net.processor import ObjectSegmentation
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# FAST API
app = FastAPI()
app.mount("/output", StaticFiles(directory="output"), name="output")

# Load the model
seg_model = U2NET(3, 1)
# model_dir = os.path.join(os.getcwd(), "u2net", "weights", "u2net_bce.pth")
# model_dir = os.path.join(os.getcwd(), "u2net", "weights", "u2net_human_seg.pth")
# model_dir = os.path.join(os.getcwd(), "u2net", "weights", "u2net_portrait.pth")
model_dir = os.path.join(os.getcwd(), "u2net", "weights", "u2net.pth")

API_HOST = config("API_HOST", default="0.0.0.0")
API_PORT = config("API_PORT", default=5000, cast=int)

if torch.cuda.is_available():
    print("Using the GPU")
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    seg_model.load_state_dict(torch.load(model_dir))
    seg_model.cuda()
else:
    print("[*] Using the CPU")
    seg_model.load_state_dict(torch.load(model_dir, map_location="cpu"))
seg_model.eval()

# init only once
object_segmentation = ObjectSegmentation(model=seg_model, batch_size=1, input_size=320)


class ImageSegmentation(BaseModel):
    image: str


@app.post("/api/image-segmentation/from-file")
async def image_segmentation_from_file(file: UploadFile = File(...), crop: bool = Form(...)):
    _start_time = time.time()
    fimage = await file.read()
    image = np.frombuffer(fimage, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # -> image is BGR

    # load data, preprocess, prediction, postprocess
    object_segmentation.load_datas(paths=None, images=[image])
    object_segmentation.preprocess()
    outputs = object_segmentation.predict()
    result = object_segmentation.postprocess(outputs, True, "output")

    foreground = result[0]["front"]
    mask = result[0]["mask"]
    trans = result[0]["trans"]
    crop_trans = result[0]["crop_trans"]

    print(f"{datetime.now()} - Processing Time: {time.time() - _start_time}")

    if crop:
        ret_url = f"http://{API_HOST}:{API_PORT}/{crop_trans}"
        return JSONResponse(status_code=200, content={"image": ret_url})
    else:
        ret_url = f"http://{API_HOST}:{API_PORT}/{trans}"
        return JSONResponse(status_code=200, content={"image": ret_url})


@app.post("/api/image-segmentation/from-base64")
async def image_segmentation_from_json(input: ImageSegmentation):
    p2s_request = jsonable_encoder(input)
    image = base64.b64decode(p2s_request["image"])
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_ANYCOLOR)

    # load data, preprocess, prediction, postprocess
    object_segmentation.load_datas(paths=None, images=[image])
    object_segmentation.preprocess()
    outputs = object_segmentation.predict()
    result = object_segmentation.postprocess(outputs, False, "output")

    foreground = result[0]["front"]
    mask = result[0]["mask"]

    res, im_png = cv2.imencode(".png", foreground)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
