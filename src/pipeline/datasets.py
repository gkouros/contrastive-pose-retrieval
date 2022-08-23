import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

# local imports
from dataloader.vkitti2_dataset import VirtualKITTIv2Dataset
from dataloader.apolloscape_dataset import ApolloScapeDataset
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset
from dataloader.kitti_object_dataset import KITTIObjectDataset
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


def get_apolloscape_train_val_test_datasets(
        root_dir,
        device,
        min_bbox_size=0,
        min_occupancy_ratio=0.0,
        subset_ratio=1.0,
        shuffle=False,
        angle_resolution=1,
        positive_type='rendering',
        random_positive=False,
        load_metadata=False,
        image_size=224,
        zero_padding=True,
        use_blacklist_dict={'train': True, 'val': True, 'test': True},
        ):

    # get transforms
    train_transforms, val_transforms = get_trainval_transforms(
            output_size=image_size, zero_padding=zero_padding)

    # create training set
    train_dataset = ApolloScapeDataset(
        root_dir=root_dir,
        split='train',
        transforms=train_transforms,
        min_bbox_size=min_bbox_size,
        min_occupancy_ratio=min_occupancy_ratio,
        subset_ratio=subset_ratio,
        shuffle=shuffle,
        angle_resolution=angle_resolution,
        random_positive=random_positive,
        positive_type=positive_type,
        horizontal_flip=True,
        bbox_expansion_factor=0.0,
        render_scale=0.5,
        load_metadata=load_metadata,
        device=device,
        use_blacklist=use_blacklist_dict['train'],
    )

    # create validation set
    val_dataset = ApolloScapeDataset(
        root_dir=root_dir,
        split='val',
        transforms=val_transforms,
        min_bbox_size=min_bbox_size,
        min_occupancy_ratio=min_occupancy_ratio,
        subset_ratio=1.0,
        shuffle=False,
        angle_resolution=angle_resolution,
        random_positive=False,
        positive_type=positive_type,
        bbox_expansion_factor=0.0,
        render_scale=0.5,
        load_metadata=load_metadata,
        device=device,
        use_blacklist=use_blacklist_dict['val'],
    )

    # create validation set
    test_dataset = ApolloScapeDataset(
        root_dir=root_dir,
        split='test',
        transforms=val_transforms,
        min_bbox_size=min_bbox_size,
        min_occupancy_ratio=min_occupancy_ratio,
        subset_ratio=1.0,
        shuffle=False,
        angle_resolution=angle_resolution,
        random_positive=False,
        positive_type=positive_type,
        bbox_expansion_factor=0.0,
        render_scale=0.5,
        load_metadata=load_metadata,
        device=device,
        use_blacklist=use_blacklist_dict['test'],
    )

    return train_dataset, val_dataset, test_dataset


def get_vkitti2_train_val_test_datasets(
        dataset_path=os.path.abspath('../datasets/vkitti2/'),
        categories=['clone', 'fog'],
        formats=['rgb', 'rgb'],
        split_type='ratio',
        split_ratio=0.9,
        train_scenes=[1, 2, 20],
        val_scenes=[6],
        test_scenes=[18],
        shuffle=False,
        ordered=True,
        min_bbox_size=2000,
        segment_instances=False,
        angle_resolution=1,
        ):

    # combine scenes for training and validation
    scenes = sorted(set(train_scenes + val_scenes + test_scenes))

    # initialize train and val indices
    train_indices = val_indices = test_indices = None
    train_subset_ratio = val_subset_ratio = test_subset_ratio = 1

    # get transforms
    train_transforms, val_transforms = \
        get_trainval_transforms(output_size=224)

    dataset = VirtualKITTIv2Dataset(
        root_dir=dataset_path,
        scenes=scenes,
        categories=categories,
        formats=formats,
        transforms=train_transforms,
        ordered=ordered,
        train=True,
        shuffle=shuffle,
        subset_ratio=1.0,
        min_bbox_size=min_bbox_size,
        segment_instances=segment_instances,
        angle_resolution=angle_resolution,
    )

    # config for splitting dataset into training and validation sets
    if split_type == 'ratio':
        train_subset_ratio = split_ratio
        val_subset_ratio = 1.0 - split_ratio / 2
        test_subset_ratio = 1.0 - split_ratio / 2
        train_scenes = val_scenes = test_scenes = scenes
        print(f'Training set contains {split_ratio*100:.0f}% of data\n'
              f'Validation set contains {val_subset_ratio*100:.0f}% of data\n'
              f'Test set contains {(1-split_ratio)*100:.0f}% of data')
    elif split_type == 'stratified':
        train_scenes = val_scenes = scenes
        indices = np.array(list(range(len(dataset))))
        labels = np.array(dataset.labels)

        # calculate count of each label
        counts = np.array([list(labels).count(x) for x in labels])
        labels = labels[~(counts == 1)]
        uniques = list(indices[counts == 1])
        indices = list(indices[~(counts == 1)])
        train_subset_ratio = split_ratio
        test_subset_ratio = val_subset_ratio = (1 - train_subset_ratio) / 2
        X_train, X_test, y_train, y_test = train_test_split(
            indices, labels, test_size=test_subset_ratio, random_state=12345,
            shuffle=True, stratify=labels)

        a = len(X_test) / len(X_train)
        X_train, X_val, _, _ = train_test_split(
            X_train, y_train, test_size=a,
            random_state=54321, shuffle=True, stratify=y_train)
        train_indices = X_train + uniques
        val_indices = X_val + uniques
        test_indices = X_test + uniques
        train_subset_ratio = val_subset_ratio = test_subset_ratio = 1.0
    elif split_type == 'scene':
        train_subset_ratio = val_subset_ratio = 1.0
        print(f'Training dataset contains scenes {train_scenes}  and '
              f'validation dataset contains scenes {val_scenes}')
    elif split_type == 'toy':
        # minimal dataset for testing pipeline
        train_subset_ratio = 1
        val_subset_ratio = 1
        train_scenes=[2]
        val_scenes=[6]
    else:
        raise ValueError(f'Invalid split_type={split_type}.'
                         'Choose between \'ratio\', \'scene\', and \'toy\'')

    # create datasets
    train_dataset = VirtualKITTIv2Dataset(
        root_dir=dataset_path,
        scenes=train_scenes,
        categories=categories,
        formats=formats,
        transforms=train_transforms,
        ordered=ordered,
        train=False if split_type == 'ratio' else True,
        indices_to_keep=train_indices,
        shuffle=shuffle,
        subset_ratio=train_subset_ratio,
        min_bbox_size=min_bbox_size,
        segment_instances=segment_instances,
        angle_resolution=angle_resolution,
    )

    val_dataset = VirtualKITTIv2Dataset(
        root_dir=dataset_path,
        scenes=val_scenes,
        categories=categories,
        formats=formats,
        transforms=val_transforms,
        ordered=ordered,
        train=False if split_type == 'ratio' else True,
        indices_to_keep=val_indices,
        shuffle=shuffle,
        subset_ratio=val_subset_ratio,
        min_bbox_size=min_bbox_size,
        segment_instances=segment_instances,
        angle_resolution=angle_resolution,
    )

    test_dataset = VirtualKITTIv2Dataset(
        root_dir=dataset_path,
        scenes=test_scenes,
        categories=categories,
        formats=formats,
        transforms=val_transforms,
        ordered=ordered,
        train=False if split_type == 'ratio' else True,
        indices_to_keep=test_indices,
        shuffle=shuffle,
        subset_ratio=test_subset_ratio,
        min_bbox_size=min_bbox_size,
        segment_instances=segment_instances,
        angle_resolution=angle_resolution,
    )
    return train_dataset, val_dataset, test_dataset

# add dataset path to pythonpath
def get_kitti_train_val_test_datasets(
        root_dir,
        rendering_dir,
        device,
        zmax=25.0,
        kitti_trainval_split_path='datasets/mv3d_kitti_splits',
        kitti_train_split='train.txt',
        kitti_val_split='val.txt',
        occlusion_level='fully_visible',
        use_hdf5=False,
        synthetic_occlusion_scale=0,
        split='train',
        object_category='car',
        transforms=None,
        subset_ratio=0.9,
        horizontal_flip=True,
        positive_type='normals',
        positive_from_db=True,
        pose_positive_threshold=1,
        use_fixed_cad_model=False,
        object_subcategory=0,
        data_from_scratch=False,
        render_from_scratch=False,
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
    train_dataset = KITTIObjectDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        zmax=zmax,
        occlusion_level=occlusion_level,
        split='train',
        transforms=train_transforms,
        subset_ratio=subset_ratio,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        device=device,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
        kitti_trainval_split_path=kitti_trainval_split_path,
        kitti_train_split=kitti_train_split,
        kitti_val_split=kitti_val_split,
    )

    # create validation set
    val_dataset = KITTIObjectDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        zmax=zmax,
        occlusion_level=occlusion_level,
        split='val',
        transforms=train_transforms,
        subset_ratio=subset_ratio,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        device=device,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
        kitti_trainval_split_path=kitti_trainval_split_path,
        kitti_train_split=kitti_train_split,
        kitti_val_split=kitti_val_split,
    )

    # create test set
    # same as val for now
    test_dataset = KITTIObjectDataset(
        root_dir=root_dir,
        rendering_dir=rendering_dir,
        zmax=zmax,
        occlusion_level=occlusion_level,
        split='test',
        transforms=train_transforms,
        subset_ratio=1.0,
        pose_positive_threshold=pose_positive_threshold,
        object_subcategory=object_subcategory,
        device=device,
        object_category=category,
        downsample_rate=downsample_rate,
        labelling_method=labelling_method,
        to_bgr=to_bgr,
        kitti_trainval_split_path=kitti_trainval_split_path,
        kitti_train_split=kitti_train_split,
        kitti_val_split=kitti_val_split,
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