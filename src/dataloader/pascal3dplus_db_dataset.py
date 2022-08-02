import os
import random
import traceback
from math import atan2, degrees, radians
from itertools import product
import numpy as np
import cv2
import pandas as pd
import torch
from tqdm.auto import tqdm
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from angles import normalize
from copy import deepcopy
from math import pi
from scipy.spatial.transform import Rotation as R
import h5py
import io
from PIL import Image
from functools import partial
import time
from sklearn.model_selection import train_test_split



# pytorch3D imports
try:
    from pytorch3d.renderer import camera_position_from_spherical_angles
except ImportError:
    print('PyTorch3D could not be imported.')
    traceback.print_exc()

# local imports
from dataloader.renderer import Renderer
from dataloader.dataset_utils import campos_to_R_T
from dataloader.pascal3dplus_constants import (
    AZIMUTH_OFFSET, ANGLE_RANGES, ANGLE_STEPS, ANGLE_DISCRETIZATIONS,
    ANGLE_LABELS, IMAGE_SIZES, CATEGORY_DISTANCES, RENDERING_FORMATS
)


class PASCAL3DPlusDBDataset(torch.utils.data.Dataset):
    """ Dataset class for the ApolloScape 3D car understanding dataset"""

    def __init__(
            self,
            root_dir,
            object_category='car',
            transforms=None,
            positive_type='rendering',
            object_subcategory=0,
            device=torch.device('cpu'),
            downsample_rate=2,
            pose_positive_threshold=1,
            labelling_method='quat',
            to_bgr=False,
            small_db=False,
            ):
        """
        Based on PyTorch Data Loading Tutorial
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            root_dir (string): Path to the root directory of the dataset
            transforms (tuple): Transforms applied on the images
            positive_type (str): Determines the format of positives
            pose_positive_threshold (bool): How close the pose should be for a
                sample to be regarded as positive
            object_subcategory (int): Train only samples matching the obj. sub.
            device (torch.device): Device to use for rendering
            downsample_rate (int): Greater than 1 for more efficient rendering
            labelling_method (str): The method for labelling the images
                - azim: Use only azimuth
                - nemo: 144 labels composed from 12 azims, 4 elevs, 3 thetas
                - euler: Each label contains euler angles [azim, elev, theta]
                - quat: Each label is an array containing a 4d quaternion
                - quat_and_model: 4d quaternion and cad model idx
            to_bgr (bool): whether to convert the images to BGR from RGB
        """
        # save arguments
        self._root_dir = root_dir
        self._object_category = object_category
        self._transforms = transforms
        self._positive_type = positive_type
        self._object_subcategory = object_subcategory
        self._device = device
        self._labelling_method = labelling_method
        self._downsample_rate = downsample_rate
        self._image_size = [x // downsample_rate for x in IMAGE_SIZES[
                            self._object_category.split('FGL')[0]]]
        self._to_bgr = to_bgr
        self._pose_positive_threshold = pose_positive_threshold
        self._small_db = small_db
        self._database_df = self.load_extended_annotation_file()
        self.labels = self.get_labels(method=self._labelling_method)
        self._idx = -1

        # open hdf5 file of DB
        if not hasattr(self, '_hf_db'):
            category = self._object_category.split('FGL')[0]
            h5path = os.path.join(
                self._root_dir, 'PASCAL3D_hdf5',
                f'{category}_{positive_type}{"_small" * self._small_db}_db.hdf5')
            logging.info(h5path)
            self._hf_db = h5py.File(h5path, 'r', libver='latest', swmr=True)

    def __len__(self):
        """ Calculates and return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self._database_df)

    def __getitem__(self, idx=None):
        # if no index is given go to next sample in queue
        idx = idx if idx is not None else (self._idx + 1) % len(self)
        self._idx = idx

        # get sample and label
        sample, label = self.read_sample(idx)

        # RGB to BGR
        if self._to_bgr:
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

        # apply transforms to anchor and positive
        if self._transforms is not None:
            sample = self._transforms(sample)
        return sample, label

    def load_extended_annotation_file(self):
        annot_path = os.path.join(
            self._root_dir, 'PASCAL3D_train_NeMo', 'annotations_csv')
        filepath = os.path.join(
            annot_path, self._object_category +
            '_small' * self._small_db + '_database_annotations.csv')
        if not os.path.exists(filepath):
            raise RuntimeError('Extended database has not been generated')
        df = pd.read_csv(filepath, index_col=False)
        # remove samples of other subcategories
        if self._object_subcategory:
            df = df[df['cad_index'] == self._object_subcategory]
        return df

    def get_label(self, idx, method='azim'):
        azim, elev, theta, _ = self.get_pose(idx)
        if method == 'azim':
            label = int(round(azim))
        elif method == 'nemo':
            label = self.get_nemo_label(azim, elev, theta)
        elif method == 'euler':
            label = np.array([azim, elev, theta])
        elif method == 'quat':
            label = R.from_euler(
                'zyx', [azim, elev, theta], degrees=False).as_quat()
        elif method == 'quat_and_model':
            qpose = R.from_euler(
                'zyx', [azim, elev, theta], degrees=False).as_quat()
            cad_idx = self._database_df.iloc[idx]['cad_index']
            label = np.insert(qpose, len(qpose), cad_idx)
        else:
            raise ValueError('Invalid labelling method selected')
        return label

    def get_labels(self, method=None):
        method = self._labelling_method if method is None else method
        azims = self._database_df['azimuth']
        elevs = self._database_df['elevation']
        thetas = self._database_df['theta']
        cad_idxs = self._database_df['cad_index']

        if method == 'azim':
            labels = azims.apply(round)
        elif method == 'nemo':
            raise NotImplementedError
        elif method == 'euler':
            labels = np.hstack((azims, elevs, thetas))
        elif method in ['quat', 'quat_and_model']:
            labels = []
            for azim, elev, theta, cad_idx in tqdm(
                    zip(azims, elevs, thetas, cad_idxs),
                    total=len(azims), desc='Converting labels to quaternions'):
                label = R.from_euler(
                    'zyx', [azim, elev, theta], degrees=False).as_quat()
                if method == 'quat_and_model':
                    cad_idx = cad_idx
                    label = np.insert(label, len(label), cad_idx)
                labels.append(label)
        else:
            raise ValueError('Invalid labelling method selected')
        return labels

    def get_pose(self, idx):
        azim = self._database_df.iloc[idx]['azimuth']
        elev = self._database_df.iloc[idx]['elevation']
        theta = self._database_df.iloc[idx]['theta']
        dist = self._database_df.iloc[idx]['distance'] * 100
        return [azim, elev, theta, dist]

    def get_positive_name_from_db_with_angles(
            self, azim, elev, theta, cad_index, sample_closest=True):
        # convert euler angles to quaternion
        qpose = R.from_euler(
            'zyx', [azim, elev, theta], degrees=False).as_quat()
        return self.get_positive_name_from_db_with_quaternion(
            qpose, cad_index, sample_closest)

    def get_positive_name_from_db_with_quaternion(
            self, qpose, cad_index, sample_closest=True):
        # filter DB dataframe according to a specific cad index
        sub_df = self._database_df[self._database_df['cad_index'] == cad_index]
        # get angles of all samples from the reduced DB dataframe
        azims = sub_df.azimuth.to_numpy()
        elevs = sub_df.elevation.to_numpy()
        thetas = sub_df.theta.to_numpy()
        euler = np.vstack((azims, elevs, thetas)).T
        # convert all angles to quaternions
        quats = R.from_euler('zyx', euler, degrees=False).as_quat()
        # calculate the distance between query and rendering poses
        distances = 2 * np.arccos(np.abs(np.dot(qpose, quats.T)))
        # select either the closest rendering or one of the closest
        if sample_closest:  # select close minimum
            closest = np.where(
                distances < radians(self._pose_positive_threshold))[0]
            # if no renderings are found below threshold pick the closest one
            if len(closest) == 0:
                idx = np.argmin(distances)
            else:
                idx = np.random.choice(closest)
        else:  # select global minimum
            idx = np.argmin(distances)
        # return name of chosen rendering
        return sub_df.iloc[idx]['name']

    def get_index(self, name):
        return self._database_df[self._database_df.name == name].index.item()

    def read_sample(self, idx, positive_type=None, labelling_method=None):
        positive_type = \
            self._positive_type if positive_type is None else positive_type
        labelling_method = \
            self._labelling_method if labelling_method is None \
            else labelling_method
        sample = self.read_rendering(idx, positive_type)
        label = self.get_label(idx, method=labelling_method)
        sample = cv2.resize(sample, self._image_size[::-1])
        return sample, label

    def read_rendering(self, idx, type_='rendering'):
        images = {
            'rendering': None,
            'depth': None,
            'normals': None,
            'silhouette': None,
            'all': None
        }
        types = list(images.keys())
        types.remove('all')
        name = self._database_df.iloc[idx]['name']
        split = ''
        dirname = 'renderings_db'
        hf = self._hf_db
        types = [type_] if type_ != 'all' else types
        for t in types:
            images[t] = self.read_h5_image(hf, split, dirname, name + '_' + t)
            images[t] = cv2.normalize(
                images[t], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        if type_ == 'all':
            for t in types:
                images[t] = cv2.bitwise_and(images[t], images['silhouette'])
                images[t] = (np.sum(images[t], axis=2) / 3).astype(np.uint8)
            channels = tuple([np.expand_dims(t, axis=-1)
                              for t in [images[t] for t in types[:3]]])
            images[type_] = np.concatenate(channels, axis=2)  # N x M x 3
        return images[type_]

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
