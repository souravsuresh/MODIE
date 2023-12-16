import numpy as np
import cv2
import random

from typing import Tuple
from torchvision.transforms.functional import resize, rotate

from PIL import ImageOps, Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import scipy.ndimage

# to_tensor = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#     ]
# )

BlurTransforms = [
    lambda img, ksize : cv2.GaussianBlur(img,(ksize,ksize),cv2.BORDER_DEFAULT), 
    lambda img, ksize : cv2.medianBlur(img,ksize if ksize<=3 else 3), 
    lambda img, ksize : cv2.blur(img,(ksize,ksize)),
    lambda img, ksize : motion_blur(img, 2*ksize+1)
]

def motion_blur(img, ksize):
    kernel = np.zeros((ksize, ksize))
    kernel[ksize//2, :] = 1
    kernel = kernel / np.sum(kernel)
    random_angle = random.randint(0, 45) #Rotate only by 45 degrees and not more
    kernel = scipy.ndimage.rotate(kernel, random_angle)
    return cv2.filter2D(img, -1, kernel)
    
def random_blur(img_to_blur, num_blurs=None, kernel_size=[1,5]):
    num = random.randint(0, num_blurs if num_blurs else len(BlurTransforms)+1)      # check this!!
    for transformation in np.random.choice(BlurTransforms, num, replace=True):
        ksize = np.random.randint(kernel_size[0], kernel_size[1])
        img_to_blur = transformation(img_to_blur, 2*ksize-1)
    return img_to_blur

def prepare_input(config, img: Image.Image, random_padding: bool = False) -> Image.Image:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        add_blur = config.add_blur
        if config.align_long_axis and (
            (config.input_size[0] > config.input_size[1] and img.width > img.height)
            or (config.input_size[0] < config.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(config.input_size))
        img.thumbnail((config.input_size[1], config.input_size[0]))
        delta_width = config.input_size[1] - img.width
        delta_height = config.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )

        img = ImageOps.expand(img, padding)

        #Apply Blurring
        if add_blur:
            orig_numpy = np.asarray(img)
            img = random_blur(orig_numpy)
            img = Image.fromarray(img)

        return img

