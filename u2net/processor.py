import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import datetime
from PIL import Image

__all__ = ["Processor"]


class ObjectSegmentation:
    def __init__(self, model, batch_size, input_size):
        # model
        self.model = model

        # image list
        self.imgs = None

        self.input_size = input_size
        self.batch_size = batch_size

        # input data
        self.input_datas = None

    # read data function
    def load_datas(self, paths, images):
        datas = []

        # read data list
        if paths is not None:
            for im_path in paths:
                assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
                im = cv2.imread(im_path)
                datas.append(im)

        if images is not None:
            datas = images

        self.imgs = datas

    # pre-processing image
    def preprocess(self):
        input_datas = []
        for image in self.imgs:
            image = cv2.resize(image, (self.input_size, self.input_size))
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)

            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

            # convert BGR to RGB
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = tmpImg[np.newaxis, :, :, :]
            input_datas.append(tmpImg)

        input_datas = np.concatenate(input_datas, 0)

        datas_num = input_datas.shape[0]
        split_num = (
            datas_num // self.batch_size + 1 if datas_num % self.batch_size != 0 else datas_num // self.batch_size
        )

        input_datas = np.array_split(input_datas, split_num)

        self.input_datas = input_datas

    def normPRED(self, d):
        ma = np.max(d)
        mi = np.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    # post-processing image
    def postprocess(self, outputs, visualization=False, output_dir="output"):
        results = []
        if visualization and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i, image in enumerate(self.imgs):
            # normalization
            pred = outputs[i, 0, :, :]

            pred = self.normPRED(pred)
            # range of pixels ~ 0-> 1

            # 320 x 320
            print("Pred Shape:", pred.shape)

            # convert torch tensor to numpy array
            h, w = image.shape[:2]
            mask = cv2.resize(pred, (w, h))  # orignal shape ~ HxW

            # ! problem with this one
            thresholded_img = self.apply_thresholding(mask.copy())
            print("Thresholded img:", thresholded_img.shape)
            xmin, ymin, xmax, ymax = self.get_bbox(thresholded_img)
            print("Bounding Box:", xmin, ymin, xmax, ymax)

            output_img = (image * mask[..., np.newaxis] + (1 - mask[..., np.newaxis]) * 255).astype(np.uint8)

            mask = (mask * 255).astype(np.uint8)

            print(output_img.shape)
            print(mask.shape)

            mask_3channels = cv2.merge((mask, mask, mask))
            print("Mask 3 channeks:", mask_3channels.shape)

            mat = cv2.bitwise_and(output_img, mask_3channels)

            tmp = cv2.cvtColor(mask_3channels, cv2.COLOR_BGR2GRAY)
            b, g, r = cv2.split(mat)

            rgba = [b, g, r, tmp]
            transparent = cv2.merge(rgba, 4)

            # ! anoteher way
            # res = np.concatenate((img_array * (mask/255), mask[:, :, [0]]), -1)
            # print(res.shape)
            # need to convert to RGB
            res = np.concatenate((output_img[:, :, ::-1], mask_3channels[:, :, [0]]), -1)
            transparent2 = Image.fromarray(res.astype("uint8"), mode="RGBA")

            crop_output_img = output_img[ymin:ymax, xmin:xmax, :]
            crop_mask_3channels = mask_3channels[ymin:ymax, xmin:xmax, :]
            crop_transparent = np.concatenate((crop_output_img[:, :, ::-1], crop_mask_3channels[:, :, [0]]), -1)
            crop_transparent2 = Image.fromarray(crop_transparent.astype("uint8"), mode="RGBA")

            if visualization:
                now = datetime.datetime.now()
                # cv2.imwrite(os.path.join(output_dir, f"result_mask_%d_{now}.png" % i), mask)
                cv2.imwrite(os.path.join(output_dir, f"result_mask_%d_{now}.png" % i), mask_3channels)
                # cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                cv2.imwrite(os.path.join(output_dir, f"result_%d_{now}.png" % i), output_img)
                # worse then PIL
                # cv2.imwrite(os.path.join(output_dir, f"result_trans_%d_{now}.png" % i), transparent)
                # chat luong > tot hon opencv
                crop_transparent_path = os.path.join(output_dir, f"result_trans_%d_{now}.png" % i)
                crop_transparent2.save(crop_transparent_path)
                transparent_path = os.path.join(output_dir, f"result_crop_trans_%d_{now}.png" % i)
                transparent2.save(transparent_path)

            # return the paths
            results.append(
                {"mask": mask, "front": output_img, "trans": transparent_path, "crop_trans": crop_transparent_path}
            )

        return results

    @staticmethod
    def apply_thresholding(img, threshold=0.9):
        """ img: HxW """
        # check the range of pixels
        # [0,1]
        if np.max(img) > 1:
            img = img / 255

        img[img > threshold] = 1
        img[img <= threshold] = 0
        return img

    # get the bounding box of the object
    @staticmethod
    def get_bbox(img):
        """ Get the bounding box of object 
            Binary image H x W, 
        """
        if len(img.shape) > 2:
            print("Shape of image is not valid")
            return

        # pickup a layer of one channel + black n white image
        # find the list of x,y
        x_starts = [
            np.where(img[i] == 1)[0][0] if len(np.where(img[i] == 1)[0]) != 0 else img.shape[0] + 1
            for i in range(img.shape[0])
        ]
        x_ends = [
            np.where(img[i] == 1)[0][-1] if len(np.where(img[i] == 1)[0]) != 0 else 0 for i in range(img.shape[0])
        ]
        y_starts = [
            np.where(img.T[i] == 1)[0][0] if len(np.where(img.T[i] == 1)[0]) != 0 else img.T.shape[0] + 1
            for i in range(img.T.shape[0])
        ]
        y_ends = [
            np.where(img.T[i] == 1)[0][-1] if len(np.where(img.T[i] == 1)[0]) != 0 else 0 for i in range(img.T.shape[0])
        ]

        # find the min
        startx = min(x_starts)
        endx = max(x_ends)
        starty = min(y_starts)
        endy = max(y_ends)

        # tuple
        start = (startx, starty)
        # tuple
        end = (endx, endy)

        return startx, starty, endx, endy

    def predict(self):
        outputs = []
        for data in self.input_datas:
            # convert numpy array to torch Float Tensor
            data = torch.from_numpy(data)
            # change the data type
            data = data.type(torch.FloatTensor)
            if torch.cuda.is_available():
                data = Variable(data.cuda())
            else:
                data = Variable(data)

            # ! replace the paddle with torch
            d1, d2, d3, d4, d5, d6, d7 = self.model(data)

            # CUDA tensor -> host memory first -> convert to numpy
            outputs.append(d1.detach().cpu().numpy())

            # model results
            outputs = np.concatenate(outputs, 0)

            return outputs
