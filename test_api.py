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
from u2net.processor import Processor
from torch.autograd import Variable

# FAST API
app = FastAPI()

# Load the model 
seg_model = U2NET(3, 1)
model_dir = os.path.join(os.getcwd(), "u2net", "weights", "u2net.pth")

if torch.cuda.is_available():
    print ("Using the GPU")
    print (torch.cuda.current_device())
    print (torch.cuda.get_device_name(0))
    seg_model.load_state_dict(torch.load(model_dir))
    seg_model.cuda()
else:
    print ("[*] Using the CPU")
    seg_model.load_state_dict(torch.load(model_dir, map_location="cpu"))
seg_model.eval()


def predict(model,input_datas):
    outputs = []
    for data in input_datas:
        # convert numpy array to torch Float Tensor
        data = torch.from_numpy(data)
        # change the data type
        data = data.type(torch.FloatTensor)
        if torch.cuda.is_available():
            data = Variable(data.cuda())
        else:
            data = Variable(data)

        # ! replace the paddle with torch
        d1, d2, d3, d4, d5, d6, d7 = model(data)

        # CUDA tensor -> host memory first -> convert to numpy
        outputs.append(d1.detach().cpu().numpy())

    outputs = np.concatenate(outputs, 0)

    return outputs

def Segmentation(images=None,
                    paths=None,
                    batch_size=1,
                    input_size=320,
                    output_dir='output',
                    visualization=False):

    # 初始化数据处理器
    processor = Processor(paths, images, batch_size, input_size)

    # 模型预测
    outputs = predict(seg_model,processor.input_datas)

    # 预测结果后处理
    results = processor.postprocess(outputs, visualization=visualization, output_dir=output_dir)

    return results

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
    result = Segmentation(
        images=[image], paths=None, batch_size=1, input_size=320, output_dir="output", visualization=False
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
    result = Segmentation(
        images=[image], paths=None, batch_size=1, input_size=320, output_dir="output", visualization=False
    )
    foreground = result[0]["front"][:, :, ::-1]
    mask = result[0]["mask"]

    res, im_png = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

if __name__ == "__main__":
    API_HOST = config("API_HOST", default="0.0.0.0")
    API_PORT = config("API_PORT",default=5000,cast=int)
    uvicorn.run(app, host=API_HOST, port=API_PORT)
