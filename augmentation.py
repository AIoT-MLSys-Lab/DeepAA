# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
from PIL import Image
import math

IMAGENET_SIZE = (224, 224) # (width, height) may set to (244, 224)

_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8 # FastAA
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def RandCrop(img, _):
    v = 4
    return mean_pad_randcrop(img, v)

def RandCutout(img, _):
    v = 16
    w, h = img.size
    x = random.uniform(0, w)
    y = random.uniform(0, h)

    x0 = int(min(w, max(0, x - v // 2)))  # clip to the range (0, w)
    x1 = int(min(w, max(0, x + v // 2)))
    y0 = int(min(h, max(0, y - v // 2)))
    y1 = int(min(h, max(0, y + v // 2)))

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def RandCutout60(img, _):
    v = 60
    w, h = img.size
    x_left = max(0, w // 2 - 256 // 2)
    x_right = min(w, w // 2 + 256 // 2)
    y_bottom = max(0, h // 2 - 256 // 2)
    y_top = min(h, h // 2 + 256 // 2)

    x = random.uniform(x_left, x_right)
    y = random.uniform(y_bottom, y_top)

    x0 = int(min(w, max(0, x - v // 2)))
    x1 = int(min(w, max(0, x + v // 2)))
    y0 = int(min(h, max(0, y - v // 2)))
    y1 = int(min(h, max(0, y + v // 2)))

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def RandFlip(img, _):
    if random.random() > 0.5:
        img = Flip(img, None)
    return img



def mean_pad_randcrop(img, v):
    # v: Pad with mean value=[125, 123, 114] by v pixels on each side and then take random crop
    assert v <= 10, 'The maximum shift should be less then 10'
    padded_size = (img.size[0] + 2*v, img.size[1] + 2*v)
    new_img = PIL.Image.new('RGB', padded_size, color=(125, 123, 114))
    new_img.paste(img, (v, v))
    top = random.randint(0, v*2)
    left = random.randint(0, v*2)
    new_img = new_img.crop((left, top, left + img.size[0], top + img.size[1]))
    return new_img

def Identity(img, v):
    return img


def RandResizeCrop_imagenet(img, _):
    # ported from torchvision
    # for ImageNet use only
    scale = (0.08, 1.0)
    ratio = (3. / 4., 4. / 3.)
    size = IMAGENET_SIZE  # (224, 224)

    def get_params(img, scale, ratio):
        width, height = img.size
        area = float(width * height)
        log_ratio = [math.log(r) for r in ratio]

        for _ in range(10):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = round(math.sqrt(target_area * aspect_ratio))
            h = round(math.sqrt(target_area / aspect_ratio))
            if 0 < w <= width and 0 < h <= height:
                top = random.randint(0, height - h)
                left = random.randint(0, width - w)
                return left, top, w, h

            # fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(ratio):
                w = width
                h = round(w / min(ratio))
            elif in_ratio > max(ratio):
                h = height
                w = round(h * max(ratio))
            else:
                w = width
                h = height
            top = (height - h) // 2
            left = (width - w) // 2
            return left, top, w, h

    left, top, w_box, h_box = get_params(img, scale, ratio)
    box = (left, top, left + w_box, top + h_box)
    img = img.resize(size=size, resample=PIL.Image.CUBIC, box=box)
    return img


def Resize_imagenet(img, size):
    w, h = img.size
    if isinstance(size, int):
        short, long = (w, h) if w <= h else (h, w)
        if short == size:
            return img
        new_short, new_long = size, int(size * long / short)
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        return img.resize((new_w, new_h), PIL.Image.BICUBIC)
    elif isinstance(size, tuple) or isinstance(size, list):
        assert len(size) == 2, 'Check the size {}'.format(size)
        return img.resize(size, PIL.Image.BICUBIC)
    else:
        raise Exception


def centerCrop_imagenet(img, _):
    # for ImageNet only
    # https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    crop_width, crop_height = IMAGENET_SIZE  # (224,224)
    image_width, image_height = img.size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)
        image_width, image_height = img.size
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

# def centerCrop_imagenet_default(img):
#     return centerCrop_imagenet(img, None)

def _parse_fill(fill, img, name="fillcolor"):
    # Process fill color for affine transforms
    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_bands:
            msg = ("The number of elements in 'fill' does not match the number of "
                   "bands of the image ({} != {})")
            raise ValueError(msg.format(len(fill), num_bands))

        fill = tuple(fill)

    return {name: fill}


def pad(img, padding_ltrb, fill=0, padding_mode='constant'):
    if isinstance(padding_ltrb, list):
        padding_ltrb = tuple(padding_ltrb)
    if padding_mode == 'constant':
        opts = _parse_fill(fill, img, name='fill')
        if img.mode == 'P':
            palette = img.getpalette()
            image = PIL.ImageOps.expand(img, border=padding_ltrb, **opts)
            image.putpalette(palette)
            return image
        return PIL.ImageOps.expand(img, border=padding_ltrb, **opts)
    elif len(padding_ltrb) == 4:
        image_width, image_height = img.size
        cropping = -np.minimum(padding_ltrb, 0)
        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, image_width - crop_right, image_height - crop_bottom))
        pad_left, pad_top, pad_right, pad_bottom = np.maximum(padding_ltrb, 0)

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)
    else:
        raise Exception

def get_mid_magnitude(l_mags):
    ops_mid_magnitude = {'Identity': None,
                         'ShearX': (l_mags - 1) // 2,
                         'ShearY': (l_mags - 1) // 2,
                         'TranslateX': (l_mags - 1) // 2,
                         'TranslateY': (l_mags - 1) // 2,
                         'Rotate': (l_mags - 1) // 2,
                         'AutoContrast': None,
                         'Invert': None,
                         'Equalize': None,
                         'Solarize': l_mags - 1,
                         'Posterize': l_mags - 1,
                         'Contrast': (l_mags - 1) // 2,
                         'Color': (l_mags - 1) // 2,
                         'Brightness': (l_mags - 1) // 2,
                         'Sharpness': (l_mags - 1) // 2,
                         'RandFlip': 'random',
                         'RandCutout': 'random',
                         'RandCutout60': 'random',
                         'RandCrop': 'random',
                         'RandResizeCrop_imagenet': 'random',
                         }
    return ops_mid_magnitude