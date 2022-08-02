import numpy as np
import cv2
import torch
import torchvision
from PIL import Image

from dataloader.synthetic_occlusion import (
    occlude_with_objects, load_occluders_file
)


class Resize(object):
    """Rescale the image in a sample to a given size.

    Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h // w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w // h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        try:
            sample = cv2.resize(
                sample, (new_h, new_w), interpolation=cv2.INTER_AREA)
        except cv2.error:
            raise ValueError(f'Invalid image size [{sample.shape}]')

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]  # original dims
        new_h, new_w = self.output_size  # new dims
        d = min(h - new_h, w - new_w) // 2  # max displacement
        dx = np.random.randint(-d, d)  # horizontal displacement
        dy = np.random.randint(-d, d)  # vertical displacement
        cx, cy = w // 2, h // 2  # original center
        new_cx, new_cy = cx + dx, cy + dy  # new center
        left = new_cx - new_w // 2
        right = new_cx + new_w // 2
        top = new_cy - new_h // 2
        bottom = new_cy + new_h // 2

        return sample[top:bottom, left:right]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis if color image/batch because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(sample.shape) == 2:
            pass
        elif len(sample.shape) == 3:
            sample = sample.transpose(2, 0, 1)
        elif len(sample.shape) == 4:
            sample = sample.transpose(0, 3, 1, 2)
        else:
            raise ValueError('Invalid array size')

        sample = (sample - np.mean(sample)) / np.std(sample)

        return torch.from_numpy(sample).float()


class ToNumpy(object):
    """Convert Tensors to ndarrays."""

    def __call__(self, sample):
        # swap color axis because
        # torch image: C X H X W
        # numpy image: H x W x C
        if len(sample.shape) == 2:
            pass
        elif len(sample.shape) == 3:
            sample = sample.permute(1, 2, 0)
        elif len(sample.shape) == 4:
            sample = sample.permute(0, 2, 3, 1)
        else:
            raise ValueError('Invalid array size')
        npsample = sample.numpy()
        npsample = (npsample - npsample.min()) \
            / (npsample.max() - npsample.min())
        return (npsample * 255.0).astype(np.uint8)
        # return sample.numpy()


class SquarePad(object):
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad)
                             for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(
            image, padding, fill=0, padding_mode='constant')


class Identity(object):
    def __call__(self, image):
        return image


class SyntheticOcclusion(object):
    def __init__(self, path, scale):
        assert 0 <= scale <= 3
        self.scale = scale
        self.occluders = load_occluders_file(path)

    def __call__(self, img):
        if isinstance(img, dict):  # for usage with NeMo
            img['img'] = Image.fromarray(occlude_with_objects(
                np.asarray(img['img']), self.occluders, self.scale))
            return img
        else:
            return occlude_with_objects(img, self.occluders, self.scale)


class BBoxNoise(object):
    def __init__(self, scale):
        self.scale = scale
        self.iou_min = 1 - scale

    def __call__(self, image):
        nemo_mode = False
        if isinstance(image, dict):  # for usage with NeMo
            sample = image
            image = np.asarray(sample['img'])
            # image = sample['img']
            nemo_mode = True

        assert np.argmin(image.shape) == 2
        h, w = image.shape[:2]
        n = round((h+w - ((h+w)**2 - 4*w*h*(1-self.iou_min)) ** 0.5) / 4)
        if n == 0:
            return image
        dx1, dy1, dx2, dy2 = np.random.randint(low=-n, high=n, size=4)
        canvas = np.zeros((h + 2*n, w + 2*n, 3), dtype=image.dtype)
        canvas[n:n+h, n:n+w] = image
        y1, x1 = (n + dy1, n + dx1)
        y2, x2 = (h + n + dy2, w + dx2)
        ret = canvas[y1:y2, x1:x2]
        h2, w2 = ret.shape[:2]
        fx = w / w2
        fy = h / h2
        if fx < fy:  # resize to w and pad to h
            ret = cv2.resize(ret, None, None, fx, fx)
            h2 = ret.shape[0]
            top = bottom = (h - h2) // 2
            extra = (h - h2) % 2
            ret = cv2.copyMakeBorder(
                ret, top, bottom + extra, 0, 0, cv2.BORDER_CONSTANT)
        else:  # resize to h and pad to w
            ret = cv2.resize(ret, None, None, fy, fy)
            w2 = ret.shape[1]
            left = right = (w - w2) // 2
            extra = (w - w2) % 2
            ret = cv2.copyMakeBorder(
                ret, 0, 0, left, right + extra, cv2.BORDER_CONSTANT)

        assert ret.shape == image.shape
        if nemo_mode:
            sample['img'] = Image.fromarray(ret.astype(np.uint8)).copy()
            return sample

        return ret
