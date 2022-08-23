import os
import numpy as np
from torchvision import transforms
from dataloader.transforms import (
    SquarePad, Identity, BBoxNoise, SyntheticOcclusion, PadResize
)


def get_trainval_transforms(
        output_size=224,
        intermediate_size=240,
        zero_padding=True,
        random_crop=True,
        bbox_noise=0,
        random_color_jitter=False,
        random_blur=False,
        occ_path=None,
        occ_scale=0,
        occ_exclude=(),
        evaluation=False,
        ):
    """
    [summary]

    Args:
        output_size (int, optional): [description]. Defaults to 224.

    Returns:
        [type]: [description]
    """

    # transforms for the samples of the training set
    train_transforms = (
        transforms.Compose(
            [
                BBoxNoise(bbox_noise) if not evaluation else Identity(),
                SyntheticOcclusion(occ_path, occ_scale) if occ_scale > 0 else Identity(),
                transforms.ToPILImage(),
                SquarePad() if zero_padding else Identity(),
                transforms.Resize(intermediate_size) if random_crop else transforms.Resize(output_size),
                transforms.RandomCrop(output_size) if random_crop else Identity(),
                transforms.ColorJitter(brightness=.5, hue=.3) if random_color_jitter else Identity(),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 1)) if random_blur else Identity(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        transforms.Compose(
            [
                transforms.ToPILImage(),
                SquarePad() if zero_padding else Identity(),
                transforms.Resize(intermediate_size) if random_crop else transforms.Resize(output_size),
                transforms.RandomCrop(output_size) if random_crop else Identity(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    )

    # transforms for the samples of the validation set
    val_transforms = (
        transforms.Compose(
            [
                BBoxNoise(bbox_noise) if evaluation else Identity(),
                transforms.ToPILImage(),
                SquarePad() if zero_padding else Identity(),
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        ),
        transforms.Compose(
            [
                transforms.ToPILImage(),
                SquarePad() if zero_padding else Identity(),
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    )

    return (train_transforms, val_transforms)


def get_inference_transforms(output_size=(256, 672)):
    inference_transforms = transforms.compose(
        PadResize(output_size),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )
    return inference_transforms
