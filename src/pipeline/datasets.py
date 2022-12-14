import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

# local imports
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset
from dataloader.pascal3dplus_dataset import IMAGE_SIZES as PASCAL_IMAGE_SIZES
from dataloader.kitti3d_dataset import KITTI3DDataset
from pipeline.transforms import get_trainval_transforms
from dataloader.synthetic_occlusion import create_occluders_file


# add dataset path to pythonpath
def get_pascal3d_train_val_test_datasets(
        root_dir,
        device,
        subset_ratio=0.8,
        positive_type='rendering',
        positive_from_db=False,
        small_db=False,
        use_fixed_cad_model=False,
        pose_positive_threshold=1,
        object_subcategory=0,
        object_category='car',
        labelling_method='azim',
        downsample_rate=2,
        use_hdf5=True,
        synthetic_occlusion_scale=0,
        weight_mode='uniform',
        augmentations=True,
        bbox_noise=0.0,
        to_bgr=False,
        evaluation=False,
        ):
    category = object_category.split('FGL')[0]
    output_size = [x // downsample_rate for x in PASCAL_IMAGE_SIZES[category]]
    test_split = 'occ' if 'FGL' in object_category else 'test'
    occluders_path = os.path.join(root_dir, 'VOCdevkit/VOC2012/')
    occ_path = os.path.join(
        occluders_path, 'occluders_without_%s.npz' % category)

    if synthetic_occlusion_scale and not os.path.exists(occ_path):
        create_occluders_file(
            occluders_path=occluders_path,
            save_path=occ_path,
            exclude=(category),
        )

    # get transforms
    train_transforms, val_transforms = get_trainval_transforms(
            output_size=output_size,
            zero_padding=False,
            random_crop=False,
            random_color_jitter=augmentations,
            random_blur=augmentations,
            occ_path=os.path.join(root_dir, occ_path),
            occ_scale=synthetic_occlusion_scale,
            occ_exclude=(object_category.split('FGL')[0]),
            bbox_noise=bbox_noise,
            evaluation=evaluation,
    )

    # create training set
    train_dataset = PASCAL3DPlusDataset(
        root_dir=root_dir,
        split='train',
        transforms=train_transforms,
        subset_ratio=subset_ratio,
        positive_type=positive_type,
        small_db=small_db,
        positive_from_db=positive_from_db,
        pose_positive_threshold=pose_positive_threshold,
        use_fixed_cad_model=use_fixed_cad_model,
        object_subcategory=object_subcategory,
        horizontal_flip=augmentations,
        device=device,
        object_category=category,
        render_from_scratch=False,
        data_from_scratch=False,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        use_hdf5=use_hdf5,
        to_bgr=to_bgr,
    )

    # create validation set
    val_dataset = PASCAL3DPlusDataset(
        root_dir=root_dir,
        split='val',
        transforms=val_transforms,
        subset_ratio=1.0 - subset_ratio,
        positive_type=positive_type,
        positive_from_db=positive_from_db,
        small_db=small_db,
        pose_positive_threshold=pose_positive_threshold,
        use_fixed_cad_model=use_fixed_cad_model,
        object_subcategory=object_subcategory,
        horizontal_flip=augmentations,
        device=device,
        object_category=category,
        render_from_scratch=False,
        data_from_scratch=False,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        use_hdf5=use_hdf5,
        to_bgr=to_bgr,
    )

    # create test set
    test_dataset = PASCAL3DPlusDataset(
        root_dir=root_dir,
        split=test_split,
        transforms=val_transforms,
        subset_ratio=1.0,
        positive_type=positive_type,
        positive_from_db=positive_from_db,
        small_db=small_db,
        pose_positive_threshold=pose_positive_threshold,
        use_fixed_cad_model=use_fixed_cad_model,
        object_subcategory=object_subcategory,
        horizontal_flip=False,
        device=device,
        object_category=object_category,
        render_from_scratch=False,
        data_from_scratch=False,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        use_hdf5=use_hdf5,
        to_bgr=to_bgr,
    )

    return train_dataset, val_dataset, test_dataset

# add dataset path to pythonpath
def get_kitti3d_train_val_test_datasets(
        root_dir,
        rendering_dir,
        occlusion_level='fully_visible',
        synthetic_occlusion_scale=0,
        object_category='car',
        transforms=None,
        subset_ratio=0.9,
        pose_positive_threshold=1,
        object_subcategory=0,
        downsample_rate=2,
        labelling_method='azim',
        weight_mode='uniform',
        augmentations=True,
        bbox_noise=0.0,
        to_bgr=False,
        evaluation=False,
        ):
    category = object_category.split('FGL')[0]
    output_size = [x // downsample_rate for x in PASCAL_IMAGE_SIZES[category]]
    occluders_path = os.path.join(rendering_dir, 'VOCdevkit/VOC2012/')
    occ_path = os.path.join(
        occluders_path, 'occluders_without_%s.npz' % category)

    if not os.path.exists(occ_path):
        create_occluders_file(
            occluders_path=occluders_path,
            save_path=occ_path,
            exclude=(category),
        )

    # get transforms
    train_transforms, val_transforms = get_trainval_transforms(
            output_size=output_size,
            zero_padding=False,
            random_crop=False,
            random_color_jitter=augmentations,
            random_blur=augmentations,
            occ_path=os.path.join(root_dir, occ_path),
            occ_scale=synthetic_occlusion_scale,
            occ_exclude=(object_category.split('FGL')[0]),
            bbox_noise=bbox_noise,
            evaluation=evaluation,
    )

    # create training set
    train_dataset = KITTI3DDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        occlusion_level=occlusion_level,
        split='train',
        transforms=train_transforms,
        subset_ratio=subset_ratio,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
    )

    # create validation set
    val_dataset = KITTI3DDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        occlusion_level=occlusion_level,
        split='val',
        transforms=train_transforms,
        subset_ratio=1.0-subset_ratio,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
    )

    # create test set
    test_dataset = KITTI3DDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        occlusion_level=occlusion_level,
        split='test',
        transforms=train_transforms,
        subset_ratio=1.0,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
    )

    return train_dataset, val_dataset, test_dataset