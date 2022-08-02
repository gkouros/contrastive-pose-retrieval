import os
import sys
import json

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('../loss_and_miner_utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

import random
from math import radians, pi
import numpy as np
import cv2
import pandas as pd
import torch
from tqdm.auto import tqdm
import logging
from matplotlib import pyplot as plt
from angles import normalize
from scipy.spatial.transform import Rotation as R
import h5py
import io
from PIL import Image
from dataloader.pascal3dplus_constants import AZIMUTH_OFFSET
from tqdm.auto import tqdm

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/loss_and_miner_utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)


class KITTI3DDataset(torch.utils.data.Dataset):
    """ Dataset class for KITTI 3D object dataset """
    def __init__(
                    self,
                    root_dir,
                    rendering_dir,
                    occlusion_level='all',
                    split='train',
                    object_category='car',
                    transforms=None,
                    pose_positive_threshold=1,
                    object_subcategory=0,
                    downsample_rate=2,
                    labelling_method='azim',
                    to_bgr=False,
                    random_seed=42,
                    subset_ratio=0.8,
                ):
        # arguments
        self._root_dir = root_dir
        self._occlusion_level = occlusion_level
        self._split = split
        self._object_category = object_category
        self._transforms = transforms
        self._pose_positive_threshold = pose_positive_threshold
        self._object_subcategory = object_subcategory
        self._labelling_method = labelling_method
        self._downsample_rate = downsample_rate
        self._to_bgr = to_bgr
        self._random_seed = random_seed
        self._subset_ratio = subset_ratio

        # baseline
        self._baseline = 0.54

        # occlusion level to value mapping
        self._occlusion_level_to_val = {'fully_visible': 0,
                                        'partly_occluded': 1,
                                        'largely_occluded': 2,
                                        'all': 3}
        assert occlusion_level in self._occlusion_level_to_val.keys()

        # set random seed
        np.random.seed(self._random_seed)

        # initialize idx
        self._idx = -1

        # get image names for current split
        actual_split = {'train': 'train', 'val': 'train', 'test': 'val'}[split]
        self._image_dir = os.path.join(root_dir, f'KITTI3D_{actual_split}_NeMo', 'images', object_category)
        self._annotation_dir = os.path.join(root_dir, f'KITTI3D_{actual_split}_NeMo', 'annotations', object_category)
        self._object_category_occ = object_category if occlusion_level == 'all' else object_category + '_' + occlusion_level
        with open(os.path.join(
                root_dir, f'KITTI3D_{actual_split}_NeMo', 'lists',
                self._object_category_occ, self._object_category_occ + '.txt'), 'r') as f:
            self._image_names = [t.strip() for t in f.readlines()]

        # keep first N or last N of samples depending on split name
        if self._split in ['train', 'test'] :
            self._image_names = self._image_names[:int(subset_ratio * len(self._image_names))]
        else:  # val split
            self._image_names = self._image_names[-int(subset_ratio * len(self._image_names)):]

        # read all labels
        self.labels = self.read_labels()

        # open hdf5 file of DB if needed
        category = self._object_category.split('FGL')[0]
        h5path = os.path.join(rendering_dir, 'PASCAL3D_hdf5', f'{category}_normals_db.hdf5')
        self._hf_db = h5py.File(h5path, 'r', libver='latest', swmr=False)
        self._rendering_annot_path = os.path.join(rendering_dir, 'PASCAL3D_train_NeMo', 'annotations_csv')

        # read annotations for renderings
        self._database_df = self.load_extended_annotation_file()

    # method to get length of data
    def __len__(self):
        return len(self._image_names)

    # method to get a crop item
    def __getitem__(self, idx):
        # get sample
        anchor, positive, label = self._get_sample(idx)

        # RGB to BGR
        if self._to_bgr:
            anchor = cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR)
            positive = cv2.cvtColor(positive, cv2.COLOR_RGB2BGR)

        # apply transforms to anchor and positive
        if self._transforms is not None:
            anchor = self._transforms[0](anchor)
            positive = self._transforms[1](positive)

        return (anchor, positive, label)

    def _get_sample(self, idx):
        # read positive
        anchor = Image.open(os.path.join(self._image_dir, self._image_names[idx]))
        anchor = np.asarray(anchor.convert('RGB') if anchor.mode != 'RGB' else anchor)
        anchor = cv2.resize(anchor, None, fx=1/self._downsample_rate, fy=1/self._downsample_rate)

        # read negative
        label = self.labels[idx]
        label = label if self._labelling_method == 'quat' else self.process_label(label, method='quat')
        positive_name = self.get_positive_name_from_db(label, True)
        positive = self.read_normals_rendering(positive_name)
        if positive.shape != anchor.shape:
            positive = cv2.resize(positive, anchor.shape[:2][::-1])

        return anchor, positive, self.labels[idx]

    def read_labels(self):
        path = os.path.join(self._annotation_dir, '../..', 'labels')
        os.makedirs(path, exist_ok=True)
        fn = f'{self._split}_occ={self._occlusion_level}_subset={self._subset_ratio}_labels.json'
        fp = os.path.join(path, fn)
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                labels = json.load(f)
            return labels

        labels = [None for _ in range(len(self))]
        for idx, image_name in tqdm(enumerate(self._image_names), total=len(self), desc=self._split + ' split'):
            name = image_name.split('.')[0]
            annotations = dict(np.load(os.path.join(self._annotation_dir, name + '.npz'), allow_pickle=True))
            label = {k: annotations[k] for k in annotations if k in ['azimuth', 'elevation', 'theta']}
            label['azimuth'] = normalize(label['azimuth'] + radians(AZIMUTH_OFFSET), 0, 2*pi)
            label['elevation'] *= -1
            label = self.process_label(label, method=self._labelling_method)
            labels[idx] = list(label)

        with open(fp, 'w') as f:
            json.dump(labels, f)

        return labels

    def process_label(self, epose, round_=False, method='azim'):
        if isinstance(epose, dict):
            azim, elev, theta = epose['azimuth'], epose['elevation'], epose['theta']
        else:
            azim, elev, theta = epose
        if method == 'azim':
            label = int(round(azim))
        elif method == 'discrete':
            label = self.get_discrete_label(azim, elev, theta)
        elif method == 'euler':
            label = np.array([azim, elev, theta])
        elif method == 'quat':
            rot = R.from_euler('zyx', [azim, elev, theta], degrees=False)
            label = rot.as_quat()
        else:
            raise ValueError('Invalid labelling method selected')
        return label

    def get_discrete_label(self, azim, elev, theta, base=10):
        return np.round(np.array([azim, elev, theta]) // base) * base

    def load_extended_annotation_file(self):
        filepath = os.path.join(self._rendering_annot_path, self._object_category + '_database_annotations.csv')
        if not os.path.exists(filepath):
            raise RuntimeError('Extended database has not been generated')
        df = pd.read_csv(filepath, index_col=False)
        # remove samples of other subcategories
        if self._object_subcategory:
            df = df[df['cad_index'] == self._object_subcategory]
        return df

    def read_normals_rendering(self, name):
        split = ''
        dirname = 'renderings_db'
        img = self.read_h5_image(self._hf_db, split, dirname, name + '_normals')
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img

    def get_positive_name_from_db(self, qpose, sample_closest=False):
        azims = self._database_df.azimuth.to_numpy()
        elevs = self._database_df.elevation.to_numpy()
        thetas = self._database_df.theta.to_numpy()
        euler = np.vstack((azims, elevs, thetas)).T
        quats = R.from_euler('zyx', euler).as_quat()
        distances = 2 * np.arccos(np.dot(qpose, quats.T))

        if sample_closest:  # select close minimum
            closest = np.where(
                distances < radians(self._pose_positive_threshold))[0]
            if len(closest) == 0:
                idx = np.argmin(distances)
            else:
                idx = np.random.choice(closest)
            name = self._database_df.iloc[idx]['name']
        else:  # select global minimum
            idx = np.argmin(distances)
            name = self._database_df.iloc[idx]['name']
        return name

    def read_h5_image(self, hf, split, dirname, name, db=False):
        loc = os.path.join(
            'train' if split == 'val' else split, dirname,
            self._object_category, f'{name}')
        data = np.array(hf[loc])
        image_stream = io.BytesIO(data)
        if 'depth' in name or 'normals' in name:
            image_stream.seek(0)
            file_bytes = np.asarray(
                bytearray(image_stream.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.asarray(Image.open(io.BytesIO(data)))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_weights(self):
        weights = [1/len(self)] * len(self)
        return weights


# main function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    dataset = KITTI3DDataset(root_dir='/esat/topaz/gkouros/datasets/KITTI3D/',
                                 rendering_dir='/esat/topaz/gkouros/datasets/pascal3d/',
                                #  occlusion_level='fully_visible',
                                #  occlusion_level='partly_occluded',
                                 occlusion_level='largely_occluded',
                                #  occlusion_level='all',
                                 split='test',
                                 labelling_method='euler',
                                 object_subcategory=1,
                                 subset_ratio=1)
    start = time.time()
    for sample in dataset:
        fig = plt.figure(figsize=(10, 10))
        # show image and print label as title
        plt.subplot(211)
        plt.imshow(sample[0])
        plt.title(sample[2])
        plt.subplot(212)
        plt.imshow(sample[1])
        # convert figure to numpy array and show
        plt.tight_layout()
        # convert to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        plt.close()
        # show image
        cv2.imshow('image', im)
        k = cv2.waitKey(0)
        if k in [27, ord('q')]:
            break