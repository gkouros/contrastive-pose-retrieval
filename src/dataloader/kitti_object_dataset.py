import os
import sys

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('../loss_and_miner_utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

import random
import traceback
from math import atan2, degrees, radians, pi
from itertools import product
import numpy as np
import cv2
import copy
import pandas as pd
import torch
from tqdm.auto import tqdm
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from angles import normalize
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import h5py
import io
from PIL import Image
from functools import partial
import time
from sklearn.model_selection import train_test_split
import pickle

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

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/loss_and_miner_utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

FRAME_SIZE = (375, 1242)

class KITTIObjectDataset(torch.utils.data.Dataset):
    """ Dataset class for KITTI 3D object dataset """
    def __init__(
                    self,
                    root_dir,
                    rendering_dir,
                    zmax=25.0,
                    occlusion_level='fully_visible',
                    split='train',
                    object_category='car',
                    transforms=None,
                    subset_ratio=0.9,
                    pose_positive_threshold=1,
                    object_subcategory=0,
                    device=torch.device('cpu'),
                    downsample_rate=2,
                    labelling_method='azim',
                    to_bgr=False,
                    random_seed=42,
                    kitti_trainval_split_path='datasets/mv3d_kitti_splits',
                    kitti_train_split='train.txt',
                    kitti_val_split='val.txt',
                ):
        """
        Args:
            root_dir (string): Root Directory of KITTI object dataset.
        """
        # arguments
        self._rendering_dir = rendering_dir
        self._occlusion_level = occlusion_level
        self._zmax = zmax
        self._split = split
        self._object_category = object_category
        self._transforms = transforms
        self._subset_ratio = subset_ratio
        self._pose_positive_threshold = pose_positive_threshold
        self._object_subcategory = object_subcategory
        self._device = device
        self._labelling_method = labelling_method
        self._downsample_rate = downsample_rate
        self._image_size = [x // downsample_rate for x in IMAGE_SIZES[
                            self._object_category.split('FGL')[0]]]
        self._to_bgr = to_bgr
        self._random_seed = random_seed
        self._kitti_trainval_split_path = kitti_trainval_split_path
        self._kitti_train_split = kitti_train_split
        self._kitti_val_split = kitti_val_split

        # paths
        if self._split in ['train', 'val']:
            self._root_dir = os.path.join(root_dir, 'training')
        else:
            self._root_dir = os.path.join(root_dir, 'testing')
        self._left_image_dir = os.path.join(self._root_dir, 'image_2')
        self._right_image_dir = os.path.join(self._root_dir, 'image_3')
        self._label_dir = os.path.join(self._root_dir, 'label_2')
        self._calib_dir = os.path.join(self._root_dir, 'calib')
        self._train_split_path = os.path.join(self._kitti_trainval_split_path, self._kitti_train_split)
        if self._kitti_val_split is not None:
            self._val_split_path = os.path.join(self._kitti_trainval_split_path, self._kitti_val_split)
        else:
            self._val_split_path = None

        # filenames
        self._frame_indices = []
        self._left_image_filenames = []
        self._right_image_filenames = []
        self._label_filenames = []
        self._left_label_dict_list = []
        self._right_label_dict_list = []
        self._K_list = []

        # a list of image crop parameters
        self._image_crop_filenames = []
        self._image_crop_labels = []

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

        # read annotations for renderings
        self._database_df = self.load_extended_annotation_file()

        # open hdf5 file of DB if needed
        category = self._object_category.split('FGL')[0]
        h5path = os.path.join(
            self._rendering_dir, 'PASCAL3D_hdf5', f'{category}_normals_db.hdf5')
        self._hf_db = h5py.File(h5path, 'r', libver='latest', swmr=False)

        # load annotations
        self.load_annotations()

        # get train and test split
        self._train_split, self._test_split = self._get_train_test_split()
        self.labels = self._get_labels()

        # print info
        # print('Number of frames found in the directory: {}'.format(len(self._left_image_filenames)))
        # print('Number of \'{}\' image crops for class \'{}\': {}'.format(occlusion_level, object_category, len(self._image_crop_filenames)))
        print('Number of \'{}\' samples for class \'{}\': {}'.format(split, object_category, self.__len__()))

    def load_annotations(self):
        fp = os.path.join(self._root_dir, f'car_{self._split}_occLevel={self._occlusion_level}_zmax={int(self._zmax)}_annotations.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                (
                    self._left_image_filenames,
                    self._right_image_filenames,
                    self._image_crop_filenames,
                    self._image_crop_labels,
                    self._frame_indices,
                    self._label_filenames,
                    self._left_label_dict_list,
                    self._right_label_dict_list,
                    self._K_list,
                ) = pickle.load(f)
                return
        # go through each files
        loop = tqdm(sorted(os.listdir(self._left_image_dir)))
        for img_fname in loop:
            loop.set_description(desc='Loading KITTI3D annotations ' + \
                f'{len(self._image_crop_labels)}')
            fname, file_ext = os.path.splitext(img_fname)

            # construct filenames
            left_image_full_path = os.path.join(self._left_image_dir, img_fname)
            right_image_full_path = os.path.join(self._right_image_dir, fname + '.png')
            label_full_path = os.path.join(self._label_dir, fname + '.txt')
            calib_full_path = os.path.join(self._calib_dir, fname + '.txt')

            if os.path.exists(left_image_full_path) and \
                    os.path.exists(right_image_full_path) and \
                    os.path.exists(calib_full_path) and \
                    os.path.exists(label_full_path):
                self._left_image_filenames.append(left_image_full_path)
                self._right_image_filenames.append(right_image_full_path)

                # get intrinsic matrix
                cam_id = 2
                with open(calib_full_path, 'r') as file:
                    calib_str = file.readlines()[cam_id].split('\n')[0]
                    if 'P2:' in calib_str:
                        calib_data = calib_str.split(': ')[-1]
                        K = np.fromstring(calib_data, dtype=np.float32, sep=' ').reshape(3, 4)

                # check if files exist
                self._label_filenames.append(label_full_path)

                # get labels
                labels_str = None
                left_label_dict_cam = []
                with open(label_full_path, 'r') as file:
                    labels_str = file.readlines()

                    # go through each object
                    for i, label in enumerate(labels_str):
                        label = label.replace('\n', '').split()
                        class_obj = label[0]
                        # check if it is a relevant class
                        if class_obj.lower() in [self._object_category]:
                            truncated = float(label[1])
                            occlusion = int(label[2]) # 0 = fully visible, 1 = partly occluded
                                                    # 2 = largely occluded, 3 = unknown
                            local_yaw = float(label[3])
                            xmin, ymin, xmax, ymax = \
                                float(label[4]), float(label[5]), float(label[6]), float(label[7])

                            h = float(label[8])
                            w = float(label[9])
                            l = float(label[10])
                            # these are in camera coordinate system
                            x_cam = float(label[11])
                            y_cam = float(label[12]) - (h / 2.0)
                            z_cam = float(label[13])
                            global_yaw = float(label[14])
                            conf = float(label[15]) if len(label) == 16 else 1

                            # check if the object is within range
                            if (z_cam < self._zmax) and conf > 0.0 and \
                                    ((self._occlusion_level == 'all') or \
                                    (occlusion <= self._occlusion_level_to_val[self._occlusion_level])):

                                label_dict = {}
                                label_dict['x'] = x_cam
                                label_dict['y'] = y_cam
                                label_dict['z'] = z_cam
                                label_dict['bbox'] = {
                                    'left': int(xmin),
                                    'top': int(ymin),
                                    'right': int(xmax),
                                    'bottom': int(ymax)}
                                if not self._verify_bbox(label_dict['bbox']):
                                    continue

                                label_dict['l'] = l
                                label_dict['w'] = w
                                label_dict['h'] = h
                                label_dict['local_yaw'] = local_yaw * 180.0 / np.pi# (local_yaw + np.pi) * 180.0 / np.pi
                                label_dict['global_yaw'] = (global_yaw + np.pi) * 180.0 / np.pi
                                label_dict['class'] = class_obj
                                label_dict['conf'] = conf
                                label_dict['occlusion'] = occlusion

                                left_label_dict_cam.append(label_dict)

                                # add to crop list
                                self._image_crop_filenames.append(left_image_full_path)
                                self._image_crop_labels.append(label_dict)
                                self._frame_indices.append(fname)

                    # get labels in right camera frame
                    right_label_dict_cam = copy.deepcopy(left_label_dict_cam)
                    for i in range(len(right_label_dict_cam)):
                        right_label_dict_cam[i]['x'] -= self._baseline

                    # append labels
                    self._left_label_dict_list.append(left_label_dict_cam)
                    self._right_label_dict_list.append(right_label_dict_cam)
                    self._K_list.append(K)

        with open(fp, 'wb') as f:
            pickle.dump(
                (
                    self._left_image_filenames,
                    self._right_image_filenames,
                    self._image_crop_filenames,
                    self._image_crop_labels,
                    self._frame_indices,
                    self._label_filenames,
                    self._left_label_dict_list,
                    self._right_label_dict_list,
                    self._K_list
                ), f)


    # method to get length of data
    def __len__(self):
        datalen = len(self._train_split['filenames']) if self._split in ['train', 'test'] else len(self._test_split['filenames'])
        return datalen

    # method to get train and test splits by random sampling
    def _get_train_test_split(self):
        # get number of frames
        num_frames = len(self._image_crop_filenames)

        if self._val_split_path is not None:
            # get validation split
            with open(self._val_split_path, 'r') as file:
                val_split_str = file.readlines()
                val_split_list = [x.split('\n')[0] for x in val_split_str]
            with open(self._train_split_path, 'r') as file:
                train_split_str = file.readlines()
                train_split_list = [x.split('\n')[0] for x in train_split_str]

            # get train and test split
            train_indices = []
            test_indices = []
            for i in range(num_frames):
                if self._frame_indices[i] in train_split_list:
                    train_indices.append(i)
                elif self._frame_indices[i] in val_split_list:
                    test_indices.append(i)
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
        else:
            # get train and test split
            train_split_len = int(num_frames * self._subset_ratio)
            test_split_len = int(num_frames * (1.0 - self._subset_ratio))

            # randomly shuffle indices
            indices = np.random.permutation(num_frames)
            train_indices = indices[:train_split_len]
            test_indices = indices[train_split_len:]

        # get train and test splits
        train_split = {'filenames':[self._image_crop_filenames[i] for i in train_indices],
                       'labels':[self._image_crop_labels[i] for i in train_indices]}
        test_split = {'filenames':[self._image_crop_filenames[i] for i in test_indices],
                      'labels':[self._image_crop_labels[i] for i in test_indices]}

        return train_split, test_split

    # method to get a sample
    def __get_sample(self, idx, camera='left'):
        assert camera in ['left', 'right']
        # get image and label
        if camera == 'left':
            image_filename = self._left_image_filenames[idx]
            label = self._left_label_dict_list[idx]
            K = self._K_list[idx]
        else:
            image_filename = self._right_image_filenames[idx]
            label = self._right_label_dict_list[idx]
            K = self._K_list[idx]

        # read image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # draw bounding boxes
        # for i in range(len(label)):
        #     xmin = int(label[i]['bbox']['left'])
        #     ymin = int(label[i]['bbox']['top'])
        #     xmax = int(label[i]['bbox']['right'])
        #     ymax = int(label[i]['bbox']['bottom'])
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # crop image 2d bounding boxes and put in a list
        image_crops = []
        for i in range(len(label)):
            xmin = label[i]['bbox']['left']
            ymin = label[i]['bbox']['top']
            xmax = label[i]['bbox']['right']
            ymax = label[i]['bbox']['bottom']
            image_crop = image[ymin:ymax, xmin:xmax, :]
            image_crops.append(image_crop)

        # construct a dict to return
        sample = {'image': image, 'image_crops': image_crops, 'label': label, 'K': K}

        return sample

    def _get_cropped_sample(self, idx):
        # read image and crop
        image_crop_filename = self._train_split['filenames'][idx] if self._split == 'train' else self._test_split['filenames'][idx]
        label = self._train_split['labels'][idx] if self._split == 'train' else self._test_split['labels'][idx]
        image_bgr = cv2.imread(image_crop_filename)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # print(label['bbox'])
        xmin, ymin, xmax, ymax = \
            round(np.clip(label['bbox']['left'], 0, FRAME_SIZE[1]-1)), \
            round(np.clip(label['bbox']['top'], 0, FRAME_SIZE[0]-1)), \
            round(np.clip(label['bbox']['right'], 0, FRAME_SIZE[1]-1)), \
            round(np.clip(label['bbox']['bottom'], 0, FRAME_SIZE[0]-1))
        image_crop = image_rgb[ymin:ymax, xmin:xmax, :]
        aspect_ratio = (xmax-xmin) / (ymax-ymin)
        target_aspect_ratio = self._image_size[::-1][0] / self._image_size[::-1][1]
        if target_aspect_ratio > aspect_ratio:
            # width need to be resized
            height = self._image_size[::-1][1]
            width = int(aspect_ratio * height)
        else:
            # height need to be resized
            width = self._image_size[::-1][0]
            height = int(width / aspect_ratio)
        # resize image
        image_crop = cv2.resize(image_crop, (width, height))
        # zero pad
        image_crop_zeropad = np.zeros((self._image_size[0], self._image_size[1], 3), np.uint8)
        # compute center offset
        x_left = (self._image_size[1] - width) // 2
        y_top = (self._image_size[0] - height) // 2
        image_crop_zeropad[y_top:y_top+image_crop.shape[0],
                           x_left:x_left+image_crop.shape[1]] = image_crop
        # get orientation label
        rpy = {'roll': 0, 'pitch': 0, 'yaw': label['local_yaw']}

        # find corresponding rendered image
        epose = np.array([radians(rpy['yaw']), 0, 0])
        qpose = R.from_euler('zyx', epose).as_quat()
        name = self.get_positive_name_from_db_with_angles(qpose, 1, False)
        image_normal = self.read_normals_rendering(name)
        # return item
        return {'anchor': image_crop_zeropad,
                'orientation_euler': rpy,
                'orientation_quat': qpose,
                'positive': image_normal}

    # method to get a crop item
    def __getitem__(self, idx):
        sample = self._get_cropped_sample(idx)
        # get label
        label = self.get_label(sample['orientation_euler'], method=self._labelling_method)
        anchor, positive = sample['anchor'], sample['positive']

        # RGB to BGR
        if self._to_bgr:
            anchor = cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR)
            positive = cv2.cvtColor(positive, cv2.COLOR_RGB2BGR)

        # apply transforms to anchor and positive
        if self._transforms is not None:
            anchor = self._transforms[0](anchor)
            positive = self._transforms[1](positive)

        return (anchor, positive, label)

    def _verify_bbox(self, bbox):
        x1, x2, y1, y2 = bbox['left'], bbox['right'], bbox['top'], bbox['bottom']
        x1 = np.clip(x1, 0, FRAME_SIZE[1])
        x2 = np.clip(x2, 0, FRAME_SIZE[1])
        y1 = np.clip(y1, 0, FRAME_SIZE[0])
        y2 = np.clip(y2, 0, FRAME_SIZE[0])
        h, w = y2 - y1, x2 - x1
        aspect_ratio = w / h
        return h > 0 and w > 0 and aspect_ratio > 0.5

    # method to get labels
    def _get_labels(self, method=None):
        data_split = self._train_split if self._split in ['train', 'test'] else self._test_split
        if method is None:
            method = self._labelling_method
        return np.array(
            [self.get_label({'roll':0,'pitch':0,'yaw':label['local_yaw']},
                            method=method) for label in data_split['labels']]
        )

    ################ PASCAL3D Functions ###############
    def get_label(self, epose, round_=False, method='azim'):
        azim, elev, theta = epose['yaw'], epose['pitch'], epose['roll']
        if method == 'azim':
            label = int(round(azim))
        elif method == 'discrete':
            label = self.get_discrete_label(azim, elev, theta) * pi / 180.0
        elif method == 'euler':
            label = np.array([azim, elev, theta]) * pi / 180.0
        elif method == 'quat':
            rot = R.from_euler('zyx', [azim, elev, theta], degrees=True)
            label = rot.as_quat()
        else:
            raise ValueError('Invalid labelling method selected')
        return label

    def get_discrete_label(self, azim, elev, theta, base=10):
        return np.round(np.array([azim, elev, theta]) // base) * base

    def load_extended_annotation_file(self):
        annot_path = os.path.join(
            self._rendering_dir, 'PASCAL3D_train_NeMo', 'annotations_csv')
        filepath = os.path.join(
            annot_path, self._object_category + '_database_annotations.csv')
        if not os.path.exists(filepath):
            raise RuntimeError('Extended database has not been generated')
        df = pd.read_csv(filepath, index_col=False)
        # remove samples of other subcategories
        if self._object_subcategory:
            df = df[df['cad_index'] == self._object_subcategory]
        return df

    def get_positive_name_from_db_with_angles(
            self, qpose, cad_index, sample_closest=True):
        # filter DB dataframe according to a specific cad index
        sub_df = self._database_df[self._database_df['cad_index'] == cad_index]
        # get angles of all samples from the reduced DB dataframe
        azims = sub_df.azimuth.to_numpy()
        elevs = sub_df.elevation.to_numpy()
        thetas = sub_df.theta.to_numpy()
        euler = np.vstack((azims, elevs, thetas)).T
        # convert all angles to quaternions
        quats = R.from_euler('zyx', euler).as_quat()
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
        # print(sub_df.iloc[idx])
        # return name of chosen rendering
        return sub_df.iloc[idx]['name']

    def read_normals_rendering(self, name):
        split = ''
        dirname = 'renderings_db'
        img = self.read_h5_image(self._hf_db, split, dirname, name + '_normals')
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img

    def get_positive_name_from_db(self, idx):
        epose = self._annot_df.iloc[idx][['azimuth', 'elevation', 'theta']
                                         ].to_numpy(dtype=np.float)
        epose[0] += radians(AZIMUTH_OFFSET)
        epose[0] = normalize(epose[0], 0, 2*pi)
        qpose = R.from_euler('zyx', epose).as_quat()
        cad_index = self._annot_df.iloc[idx]['cad_index']
        sub_df = self._database_df[self._database_df['cad_index'] == cad_index]
        azims = sub_df.azimuth.to_numpy()
        elevs = sub_df.elevation.to_numpy()
        thetas = sub_df.theta.to_numpy()
        euler = np.vstack((azims, elevs, thetas)).T
        quats = R.from_euler('zyx', euler).as_quat()
        distances = 2 * np.arccos(np.dot(qpose, quats.T))

        if True:  # select close minimum
            closest = np.where(
                distances < radians(self._pose_positive_threshold))[0]
            if len(closest) == 0:
                idx = np.argmin(distances)
            else:
                idx = np.random.choice(closest)
            name = sub_df.iloc[idx]['name']
        else:  # select global minimum
            idx = np.argmin(distances)
            name = sub_df.iloc[idx]['name']
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

    dataset = KITTIObjectDataset(root_dir='/media/omega/ShubDatasets/KITTI/object/',
                                 rendering_dir='./NeMo/data',
                                 occlusion_level='fully_visible',
                                 split='train')
    for sample in dataset:
        fig = plt.figure(figsize=(10, 10))
        # show image and print label as title
        plt.subplot(211)
        plt.imshow(sample['anchor'])
        plt.title(sample['label_euler'])
        plt.subplot(212)
        plt.imshow(sample['positive'])
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
        cv2.waitKey(0)