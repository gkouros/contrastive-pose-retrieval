import cv2
from matplotlib import pyplot as plt
import torch
from dataloader.dataset_utils import tensor2numpyUInt8


def plot_imgs(imgs, title=None, horizontally=True, normalize=False):
    if horizontally:
        fig, ax = plt.subplots(1,len(imgs))
        fig.set_size_inches([10, 5])
    else:
        fig, ax = plt.subplots(len(imgs), 1)
        fig.set_size_inches([20, 10])
    for i, img in enumerate(imgs):
        if normalize:
            img = cv2.normalize(img, None, alpha=0, beta=255,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ax[i].imshow(img)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    if title is not None:
        fig.suptitle(title)


def plot_anchor_positive_pair(anchor, positive, label=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    if isinstance(anchor, int):
        anchor = tensor2numpyUInt8(anchor)
    if isinstance(anchor, int):
        positive = tensor2numpyUInt8(positive)
    ax[0].imshow(anchor)
    ax[1].imshow(positive)


def plot_anchor_positive_blended(
        anchor, positive, label='nan', figsize=(30, 20)):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(anchor)
    ax[1].imshow(positive)
    ax[2].imshow(cv2.addWeighted(anchor, 1, positive, 0.8, 0))
    ax[0].set_title(f'anchor - label={label}')
    ax[1].set_title(f'positive - label={label}')
    ax[2].set_title(f'blended - label={label}')
