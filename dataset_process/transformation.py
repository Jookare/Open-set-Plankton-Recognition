from torch import nn
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as v2F
from scipy.stats import mode


class SYKE2024_zoo_transform(nn.Module):
    def __init__(self, image_size, crop=True):
        super(SYKE2024_zoo_transform, self).__init__()
        self.image_size = image_size
        self.crop = crop

    def forward(self, img):
        img = img.convert("L")
        fill = find_dominant_color(img)
        transform_list = [ResizeFill(self.image_size, fill)]
        
        if self.crop:
            transform_list.insert(0, CropFromBottom(25))
            
        if self.training:
            transform_list.extend([
                v2.RandomPerspective(distortion_scale=0.1, p=0.5, fill=fill),
                v2.RandomAffine(
                    degrees=(-90, 90),
                    translate=(0, 0.1),
                    scale=(0.95, 1.01),
                    shear=(1, 1.01),
                    fill=fill,
                )]
            )
        transform = v2.Compose(transform_list)
        return transform(img)
    
class SYKE2022_phyto_transform(nn.Module):
    def __init__(self, image_size):
        super(SYKE2022_phyto_transform, self).__init__()
        self.image_size = image_size

    def forward(self, img):
        img = img.convert("L")
        fill = find_dominant_color(img)
        transform_list = [
            ResizeFill(self.image_size, fill),
        ]
        
        if self.training:
            transform_list.extend([
                v2.RandomPerspective(distortion_scale=0.05, p=0.5, fill=fill),
                v2.RandomAffine(
                    degrees=(-90, 90),
                    translate=(0, 0.1),
                    scale=(0.95, 1.05),
                    shear=(1, 1.05),
                    fill=fill,
                )]
            )
        transform = v2.Compose(transform_list)
        return transform(img)


class ResizeFill(nn.Module):

    def __init__(self, size, fill):
        super(ResizeFill, self).__init__()
        self.size = size
        self.fill = fill

    def forward(self, img):
        ops = ImageOps.contain(img, self.size)
        out = Image.new(img.mode, self.size, self.fill)
        # Paste to the center of the image
        out.paste(ops, ((self.size[0] - ops.size[0]) // 2, (self.size[1] - ops.size[1]) // 2))
        return out


def find_dominant_color(image_pil):
    """
    Finds the most dominant color (mode) from image.
    Can be then used to fill background.
    """

    return (
        tuple(mode(mode(image_pil).mode).mode.tolist())
        if image_pil.mode == "RGB"
        else int(mode(image_pil, axis=None, keepdims=False).mode)
    )


class CropFromBottom(nn.Module):

    def __init__(self, p):
        super(CropFromBottom, self).__init__()
        self.p = p

    def forward(self, img):
        width, height = img.size
        img = v2F.crop(img, 0, 0, height - self.p, width)
        return img
