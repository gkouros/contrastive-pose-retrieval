import os
import random
import traceback
from math import atan2, degrees, radians, pi
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


class PASCAL3DPlusDataset(torch.utils.data.Dataset):
    """ Dataset class for the ApolloScape 3D car understanding dataset"""

    def __init__(
            self,
            root_dir,
            use_hdf5=True,
            split='train',
            object_category='car',
            transforms=None,
            subset_ratio=1.0,
            horizontal_flip=True,
            positive_type='rendering',
            positive_from_db=False,
            small_db=False,
            pose_positive_threshold=1,
            use_fixed_cad_model=False,
            object_subcategory=0,
            device=torch.device('cpu'),
            data_from_scratch=False,
            render_from_scratch=False,
            downsample_rate=2,
            labelling_method='azim',
            weight_mode='uniform',
            to_bgr=False,
            ):
        """
        Based on PyTorch Data Loading Tutorial
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            root_dir (string): Path to the root directory of the dataset
            use_hdf5 (bool): Read images from hdf5 file if true
            split (str): determines whether to load train or val split
            transforms (tuple): Transforms applied on the images
            subset_ratio (float): The ratio of data to use
            horizontal_flip (bool): Use horizontal flip for data augmentation
            positive_type (str): Determines the format of positives
            positive_from_db (bool): Get positives from extended database
            small_db(bool): Use 178k db instead of 889k db
            pose_positive_threshold (bool): How close the pose should be for a
                sample to be regarded as positive
            use_fixed_cad_model (int): If true use same model for all samples
            object_subcategory (int): Train only samples matching the obj. sub.
            device (torch.device): Device to use for rendering
            data_from_scratch: Creates annotation and mesh file from scratch
            render_from_scratch: Render positives rather than read from file
            downsample_rate (int): Greater than 1 for more efficient rendering
            labelling_method (str): The method for labelling the images
                - azim: Use only azimuth discretized with a step of 1 degree
                - discrete: Discretized 3d euler angles
                - euler: Each label contains euler angles [azim, elev, theta]
                - quat: Each label is an array containing a 4d quaternion
                - quat_and_model: 4d quaternion and cad model idx
            weight_mode (str): method of weighting for sampling
            to_bgr (bool): whether to convert the images to BGR from RGB
        """
        # save arguments
        self._root_dir = root_dir
        self._use_hdf5 = use_hdf5
        self._split = split
        self._object_category = object_category
        self._transforms = transforms
        self._subset_ratio = subset_ratio
        self._horizontal_flip = horizontal_flip
        self._positive_type = positive_type
        self._positive_from_db = positive_from_db
        self._small_db = small_db
        self._pose_positive_threshold = pose_positive_threshold
        self._use_fixed_cad_model = use_fixed_cad_model
        self._object_subcategory = object_subcategory
        self._device = device
        self._data_from_scratch = data_from_scratch
        self._render_from_scratch = render_from_scratch
        self._labelling_method = labelling_method
        self._downsample_rate = downsample_rate
        self._image_size = [round(x / downsample_rate) for x in IMAGE_SIZES[
                            self._object_category.split('FGL')[0]]]
        self._to_bgr = to_bgr
        self._weight_mode = weight_mode

        # additional auxiliary  variables
        if split in ['train', 'val', 'db']:
            self._data_dir = 'PASCAL3D_train_NeMo'
        elif split == 'test':
            self._data_dir = 'PASCAL3D_NeMo'
        elif split == 'occ':
            self._data_dir = 'PASCAL3D_OCC_NeMo'
            if not ('FGL' in object_category and 'BGL' in object_category):
                raise ValueError(
                    'Selected occluded split but object_category does not '
                    'contain one of FGL1_BGL1|FGL2_BGL2|FGL3_BGL3')
        else:
            raise ValueError('split should be one of train|val|test|occ')

        self._annot_df = self.load_annotation_file()
        if self._positive_from_db:
            self._database_df = self.load_extended_annotation_file()
        self.labels = self.get_labels(method=self._labelling_method)
        self._cad_models = self.load_cad_models_from_file()
        self._idx = -1
        render_size = int(max(self._image_size))
        self._renderer = Renderer(self._device, None, (render_size, ) * 2,
                                  dataset='pascal3d')
        # open hdf5 file of dataset if needed
        if self._use_hdf5 and not hasattr(self, '_hf'):
            category = self._object_category.split('FGL')[0]
            h5path = os.path.join(
                self._root_dir, 'PASCAL3D_hdf5', f'{category}.hdf5')
            self._hf = h5py.File(h5path, 'r', libver='latest', swmr=True)

        # open hdf5 file of DB if needed
        if self._positive_from_db and not hasattr(self, '_hf_db'):
            category = self._object_category.split('FGL')[0]
            suffix = 'small_db' if small_db else 'db'
            h5path = os.path.join(
                self._root_dir, 'PASCAL3D_hdf5',
                f'{category}_{positive_type}_{suffix}.hdf5')
            self._hf_db = h5py.File(h5path, 'r', libver='latest', swmr=True)

    def __len__(self):
        """ Calculates and return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self._annot_df)

    def __getitem__(self, idx=None):
        # if no index is given go to next sample in queue
        idx = idx if idx is not None else (self._idx + 1) % len(self)
        self._idx = idx

        # read anchor positive image pair and label
        anchor, positive, label = self.get_sample_pair(idx, self._positive_type)

        # RGB to BGR
        if self._to_bgr:
            anchor = cv2.cvtColor(anchor, cv2.COLOR_RGB2BGR)
            positive = cv2.cvtColor(positive, cv2.COLOR_RGB2BGR)

        # apply transforms to anchor and positive
        if self._transforms is not None:
            anchor = self._transforms[0](anchor)
            positive = self._transforms[1](positive)

        return (anchor, positive, label)

    def load_annotation_file(self):
        annot_path = os.path.join(
            self._root_dir, self._data_dir, 'annotations_csv')
        if not os.path.exists(annot_path):
            os.makedirs(annot_path)
        filepath = os.path.join(
            annot_path, self._object_category + '_annotations.csv')
        # load annotation file or create from scratch if not exists
        if not self._data_from_scratch and os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=False)
        else:
            df = self.load_annotations(save_path=filepath)
        # remove samples of other subcategories
        if self._object_subcategory:
            df = df[df['cad_index'] == self._object_subcategory]
        # double dataset by adding horizontally flipped images if flag is set
        df['flip'] = False
        if self._horizontal_flip:
            df = self.append_horizontally_flipped_samples(df)

        # keep only a subset of the dataset using stratified sampling
        if 0.0 < self._subset_ratio < 1.0:
            indices = list(range(len(df)))
            labels = df['cad_index'].values
            if self._split == 'train':
                indices, _, _, _ = train_test_split(
                    indices, labels, test_size=1.0-self._subset_ratio,
                    random_state=12345, shuffle=True, stratify=labels)
            elif self._split == 'val':
                _, indices, _, _ = train_test_split(
                    indices, labels, test_size=self._subset_ratio,
                    random_state=12345, shuffle=True, stratify=labels)
            df = df.iloc[indices]

        return df

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

    def load_annotations(self, save_path=None):
        attributes = ['name', 'azimuth', 'elevation', 'theta',
                      'distance', 'cad_index']
        annot_path = os.path.join(self._root_dir, self._data_dir,
                                  'annotations', self._object_category, '%s')
        annot_dict = {key: [] for key in attributes}
        # load annotations for all images
        image_names = sorted(os.listdir(os.path.join(
            self._root_dir, self._data_dir, 'images', self._object_category)))
        image_names = [name.split('.')[0] for name in image_names]
        loop = tqdm(image_names, desc='Loading annotations')
        for name in loop:
            annot_file = np.load(annot_path % name + '.npz', allow_pickle=True)
            for key in annot_dict:
                annot_dict[key].append(annot_file[key].item())
        # convert dictionary to dataframe
        df = pd.DataFrame.from_dict(annot_dict)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def append_horizontally_flipped_samples(self, df):
        flipped_df = deepcopy(df)
        flipped_df['flip'] = True
        flipped_df['azimuth'] = \
            flipped_df['azimuth'].apply(lambda x: normalize(pi - x, 0, 2 * pi))
        flipped_df['theta'] = flipped_df['theta'].mul(-1)
        return pd.concat((df, flipped_df), ignore_index=True)

    def load_cad_models_from_file(self):
        models_path = os.path.join(self._root_dir, self._data_dir, 'cad_models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        path = os.path.join(
            models_path, f'{self._object_category}_cad_models.npy')
        if not self._data_from_scratch and os.path.exists(path):
            models = np.load(path, allow_pickle=True)
        else:
            models = self.load_cad_models(save_path=path)
        return models

    def load_cad_models(self, save_path=None):
        models = []
        models_path = os.path.join(
            self._root_dir, 'PASCAL3D+_release1.1', 'CAD',
            self._object_category.split('FGL')[0])
        model_names = sorted(os.listdir(models_path))
        for model_name in tqdm(model_names, desc='Loading CAD models'):
            fpath = os.path.join(models_path, model_name)
            with open(fpath, 'rb') as f:
                f.readline().strip()  # ignore description
                n_verts, n_faces, _ = tuple(
                    [int(s) for s in f.readline().strip().split(b' ')])
                verts = np.array(
                    [[float(s) for s in f.readline().strip().split(b' ')]
                     for i_vert in range(n_verts)])
                faces = np.array(
                    [[int(s) for s in f.readline().strip().split(b' ')][1:]
                     for i_face in range(n_faces)])
                if len(verts) != n_verts or len(faces) != n_faces:
                    raise Exception('Failed to load models')
                models.append(dict(name=model_name, vertices=verts, faces=faces))
        np.save(save_path, models)
        return models

    def get_label(self, idx, method='azim'):
        azim, elev, theta, _ = self.get_pose(idx)
        if method == 'azim':
            label = int(round(azim))
        elif method == 'discrete':
            label = self.get_discrete_label(azim, elev, theta)
        elif method == 'euler':
            label = np.array([azim, elev, theta])
        elif method == 'quat':
            label = R.from_euler(
                'zyx', [azim, elev, theta], degrees=False).as_quat()
        elif method == 'quat_and_model':
            qpose = R.from_euler(
                'zyx', [azim, elev, theta], degrees=False).as_quat()
            cad_idx = self._annot_df.iloc[idx]['cad_index']
            label = np.insert(qpose, len(qpose), cad_idx)
        else:
            raise ValueError('Invalid labelling method selected')
        return label

    def get_discrete_label(self, azim, elev, theta, base=radians(10)):
        # def digitize(x, y):
        #     bin = (x - ANGLE_RANGES[y][0]) / ANGLE_STEPS[y]
        #     return int(round(bin))
        # qazim = digitize(azim, 'azimuth')
        # qazim = qazim if qazim != ANGLE_DISCRETIZATIONS['azimuth'] else 0
        # qelev = digitize(elev, 'elevation')
        # qtheta = digitize(theta, 'theta')
        # label = ANGLE_LABELS[(qazim, qelev, qtheta)]
        # return label
        return np.round(np.array([azim, elev, theta]) / base) * base

    def get_labels(self, method=None):
        if method is None:
            method = self._labelling_method
        return np.array(
            [self.get_label(idx, method=method) for idx in range(len(self))]
        )

    def get_cad_occurrences(self):
        return self._annot_df['cad_index'].value_counts().sort_index()

    def get_cad_frequencies(self):
        return self.get_cad_occurrences() / len(self)

    def get_weights(self):
        assert self._weight_mode in ['uniform', 'cad']
        if self._weight_mode == 'cad':
            cad_weights = self.get_cad_frequencies()
            cad_occurrences = self.get_cad_occurrences()
            weights = self._annot_df['cad_index'].apply(
                lambda x: cad_weights[x] / cad_occurrences[x]).values
        else:
            weights = [1/len(self)] * len(self)
        return weights

    def get_pose(self, idx):
        azim = self._annot_df.iloc[idx]['azimuth']
        sign = -1 if self._annot_df.iloc[idx]['flip'] else 1
        azim = normalize(azim + sign * radians(AZIMUTH_OFFSET), 0, 2 * np.pi)
        elev = self._annot_df.iloc[idx]['elevation']
        theta = self._annot_df.iloc[idx]['theta']
        dist = self._annot_df.iloc[idx]['distance'] * 100
        return [azim, elev, theta, dist]

    def get_sample_pair(self, idx, positive_type='rendering',
                        labelling_method=None):
        # read images
        anchor = self.read_camera_image(idx)
        positive = self.read_rendering(idx, self._positive_type)

        # flip images if flag is set
        if self._annot_df.iloc[idx]['flip']:
            anchor = anchor[:, ::-1]
            if not self._positive_from_db and not self._render_from_scratch:
                # to flip the normals, you have to flip the red channel
                if positive_type == 'normals':
                    try:
                        mask = self.read_rendering(idx, 'silhouette')[..., 0]
                        positive[..., 0] = (255 - positive[..., 0]) & mask
                    except Exception:
                        logging.warn('No silhouettes are available for '
                                     'flipping x axis of normals')

                # flip the image horizontally
                positive = positive[:, ::-1]

        # resize images
        anchor = cv2.resize(anchor, self._image_size[::-1])
        positive = cv2.resize(positive, self._image_size[::-1])

        # get label
        label = self.get_label(idx, method=self._labelling_method if
                               labelling_method is None else labelling_method)
        return (anchor, positive, label)

    def read_camera_image(self, idx):
        name = self._annot_df.iloc[idx]['name']
        split = 'train' if self._split == 'val' else self._split
        if self._use_hdf5:
            image = self.read_h5_image(
                self._hf, split, 'images', self._object_category, name)
        else:
            path = os.path.join(self._root_dir, self._data_dir, 'images',
                                self._object_category, name + '.JPEG')
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

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
        if self._render_from_scratch or not self.verify_ith_renderings(idx):
            images['rendering'], images['silhouette'], \
                images['depth'], images['normals'] = self.render(idx)
        else:
            category = self._object_category.split('FGL')[0]
            if self._positive_from_db:
                name = self.get_positive_name_from_db(idx)
                split = ''
                dirname = 'renderings_db'
                hf = self._hf_db if self._use_hdf5 else None
                path = os.path.join(
                    self._root_dir, 'PASCAL3D_train_NeMo',
                    'renderings_extended', category, name + '_%s')
            else:
                name = self._annot_df.iloc[idx]['name']
                if self._split == 'val':
                    split = 'train'
                elif self._split == 'occ' and self._use_fixed_cad_model:
                    split = 'test'
                else:
                    split = self._split
                dirname = \
                    self._use_fixed_cad_model * 'single_cad_' + 'renderings'
                hf = self._hf if self._use_hdf5 else None
                path = os.path.join(
                    self._root_dir, self._data_dir,
                    'single_cad_' * self._use_fixed_cad_model + 'renderings',
                    category, name + '_%s')

            types = [type_] if type_ != 'all' else types
            for t in types:
                if self._use_hdf5:
                    images[t] = self.read_h5_image(
                        hf, split, dirname, category, name + '_' + t)
                else:
                    suffix = t + '.' + RENDERING_FORMATS[t]
                    images[t] = cv2.imread(path % suffix, -1)
                    images[t] = cv2.cvtColor(images[t], cv2.COLOR_BGR2RGB)
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

    def read_h5_image(self, hf, split, dirname, category, name, db=False):
        loc = os.path.join(
            'train' if split == 'val' else split, dirname, category, f'{name}')
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

    def get_positive_name_from_db(self, idx):
        # get euler angles of the query sample from the PASCAL3D dataframe
        epose = self._annot_df.iloc[idx][['azimuth', 'elevation', 'theta']
                                         ].to_numpy(dtype=np.float)
        # correct azimuths so that 0 corresponds to right facing objects
        if self._annot_df.iloc[idx]['flip']:
            epose[0] -= radians(AZIMUTH_OFFSET)
        else:
            epose[0] += radians(AZIMUTH_OFFSET)
        # normalize azimuth to range 0-2pi
        epose[0] = normalize(epose[0], 0, 2*pi)
        # get cad model index of query sample
        cad_index = self._annot_df.iloc[idx]['cad_index']
        # find the closest or one of the closest samples from the renderings DB
        return self.get_positive_name_from_db_with_angles(
            epose[0], epose[1], epose[2], cad_index)

    def get_positive_name_from_db_with_angles(
            self, azim, elev, theta, cad_index, sample_closest=True):
        # convert euler angles to quaternion
        qpose = R.from_euler('zyx', [azim, elev, theta], degrees=False).as_quat()

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
        distances = 2 * np.arccos(np.dot(qpose, quats.T))
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

    def get_cad_model(self, cad_idx):
        cad_idx = 1 if self._use_fixed_cad_model else cad_idx
        cad_model = deepcopy(self._cad_models[cad_idx-1])
        # adjust for coordinate system differences between pytorch3d and pascal

        def pre_process_mesh_pascal(v):
            return np.hstack((v[:, 0:1], v[:, 2:3], -v[:, 1:2]))

        cad_model['vertices'] = pre_process_mesh_pascal(cad_model['vertices'])
        return cad_model

    def render(self, idx):
        cad_idx = self._annot_df.iloc[idx]['cad_index']
        cad_model = self.get_cad_model(cad_idx)
        # get angles and distance of object
        azim, elev, theta, dist = self.get_pose(idx)
        azim -= radians(AZIMUTH_OFFSET)  # left facing corresponds to 0 degrees
        # azim *= -1 if self._annot_df.iloc[idx]['flip'] else 1
        return self.generate_rendering(cad_model, azim, elev, theta, dist)

    def generate_rendering(self, cad_model, azim, elev, theta, dist):
        """
        Generates a rendering given cad model and pose

        Details:
            Expects the pose to be in degrees
        """
        # convert spherical angles to camera position
        category = self._object_category.split('FGL')[0]
        C = camera_position_from_spherical_angles(
            CATEGORY_DISTANCES[category], elev, azim,
            # dist / 100, elev, azim,
            degrees=False, device=self._device)
        theta = torch.tensor(theta, dtype=torch.float).to(self._device)
        R, T = campos_to_R_T(C, theta, device=self._device)
        # get transform for lights
        tf = (tuple(torch.matmul(
            -R[0], T[0].view(-1, 1)).cpu().numpy().T[0].tolist()),)
        # render images in RGB NOT BGR!!!
        rendering, silhouette, depth, normals = self._renderer.renderRT(
            cad_model, R, T, light_location=tf, type_=self._positive_type)

        # adjust dimensions
        w = h = max(self._image_size)
        new_h, new_w = self._image_size
        center = [h // 2, w // 2]
        x = int(center[1] - new_w / 2)
        y = int(center[0] - new_h / 2)
        if self._positive_type in ['rendering', 'all']:
            rendering = rendering[0][y:y+new_h, x:x+new_w]
            rendering = cv2.normalize(
                rendering, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if self._positive_type in ['silhouette', 'all']:
            silhouette = silhouette[0][y:y+new_h, x:x+new_w]
            silhouette = cv2.normalize(
                silhouette, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if self._positive_type in ['depth', 'all']:
            depth = depth[0][y:y+new_h, x:x+new_w]
            depth = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if self._positive_type in ['normals', 'all']:
            normals = normals[0][y:y+new_h, x:x+new_w]
            normals = cv2.normalize(
                normals, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return (rendering, silhouette, depth, normals)

    def verify_renderings(self):
        count = 0
        rng = range(len(self._annot_df) // (2 if self._horizontal_flip else 1))
        loop = tqdm(rng)
        for idx in loop:
            count += int(not self.verify_ith_renderings(idx))
            loop.set_description(f'Missing: {count}')
        if count > 0:
            logging.warn(f' {count} renderings are missing')
            return False
        else:
            logging.info('Renderings verified succesfully.')
            return True

    def verify_ith_renderings(self, idx):
        name = self._annot_df.iloc[idx]['name']
        category = self._object_category.split('FGL')[0]
        if self._use_hdf5:
            if self._split == 'val':
                split = 'train'
            elif self._split == 'occ':
                split = 'test'
            else:
                split = self._split
            images = self._hf[os.path.join(
                split,
                'single_cad_' * self._use_fixed_cad_model + 'renderings',
                category if self._use_fixed_cad_model else self._object_category)]
            for type_ in ['rendering', 'silhouette', 'depth', 'normals']:
                exists = f'{name}_{type_}' in images.keys()
                if self._positive_type in [type_, 'all'] and not exists:
                    return False
        else:
            fpath = os.path.join(
                self._root_dir, self._data_dir,
                'single_cad_' * self._use_fixed_cad_model + 'renderings',
                category, name + '_%s.%s')
            for type_, format_ in zip(
                    ['rendering', 'silhouette', 'depth', 'normals'],
                    ['JPEG', 'JPEG', 'TIFF', 'TIFF']):
                exists = os.path.exists(fpath % (type_, format_))
                if (self._positive_type in [type_, 'all'] and not exists):
                    return False
        return True

    def render_and_save(self, skip_existing=True):
        rng = range(len(self._annot_df) // (2 if self._horizontal_flip else 1))
        for idx in tqdm(rng):
            name = self._annot_df.iloc[idx]['name']
            path = os.path.join(
                self._root_dir, self._data_dir,
                'single_cad_' * self._use_fixed_cad_model + 'renderings',
                self._object_category)
            if not os.path.exists(path):
                os.makedirs(path)
            fpath = os.path.join(path, name + '_%s.%s')
            missing = False
            # check if missing
            for type_, format_ in zip(
                    ['rendering', 'silhouette', 'depth', 'normals'],
                    ['JPEG', 'JPEG', 'TIFF', 'TIFF']):
                if not os.path.exists(fpath % (type_, format_)):
                    missing = True
                    break
            # save renderings
            if self._render_from_scratch or missing:
                # render images in RGB NOT BGR!!!
                rendering, silhouette, depth, normals = self.render(idx)
                if self._positive_type in ['rendering', 'all']:
                    rendering = cv2.normalize(
                        rendering, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    cv2.imwrite(fpath % ('rendering', 'JPEG'), rendering)
                if self._positive_type in ['silhouette', 'all']:
                    silhouette = cv2.normalize(
                        silhouette, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
                    cv2.imwrite(fpath % ('silhouette', 'JPEG'), silhouette)
                if self._positive_type in ['depth', 'all']:
                    cv2.imwrite(fpath % ('depth', 'TIFF'), depth)
                if self._positive_type in ['normals', 'all']:
                    normals = cv2.normalize(
                        normals, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # normals is generated in RGB and imwrite saves in BGR so
                    # the channels must be reversed
                    normals = cv2.cvtColor(normals, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(fpath % ('normals', 'TIFF'), normals)

    def render_extended_db(self, skip_existing=True, name='db'):
        assert not ('FGL' in self._object_category)

        # initialize attribute ranges
        num_azims = 72 if self._small_db else 360
        azim_range = np.linspace(0, 2 * pi, num_azims, endpoint=False)
        elev_range = np.linspace(-pi / 6, pi / 3, 19, endpoint=True)
        theta_range = np.linspace(-pi / 6, pi / 6, 13, endpoint=True)
        cad_range = range(1, len(self._cad_models)+1)
        cad_range = [1] if self._use_fixed_cad_model else cad_range
        num_items = len(azim_range) * len(elev_range) * len(theta_range) \
            * len(cad_range)
        num_digits = len(str(num_items))
        # 360 x 19 x 13 x 10 = 889,200
        items = product(azim_range, elev_range, theta_range, cad_range)
        logging.info('%s.%s dataset size: %d' % (
            self._split, self._object_category, num_items))

        # create annotation dictionary
        attributes = ['name', 'azimuth', 'elevation', 'theta',
                      'distance', 'cad_index']
        annot_dict = {key: [] for key in attributes}

        # create path to hdf5 file
        path = os.path.join(self._root_dir, 'PASCAL3D_hdf5')
        if not os.path.exists(path):
            os.makedirs(path)
        h5path = os.path.join(
            path, f'{self._object_category}_{self._positive_type}_{name}.hdf5')

        # open file and create groups
        start = 0
        with h5py.File(h5path, 'a', libver='latest') as hf:
            if 'renderings_db' in hf.keys():
                split_group = hf['/renderings_db']
            else:
                split_group = hf.create_group('/renderings_db')
            if self._object_category in split_group.keys():
                cat_group = split_group[self._object_category]
            else:
                cat_group = split_group.create_group(self._object_category)
            if len(cat_group.keys()) > 0:
                last_item_name = sorted(cat_group.keys())[-1]
                start = int(last_item_name[1:].split('_')[0])

        logging.info('Starting from idx = %d' % start)
        it = enumerate(items)
        initial = 0

        while True:
            # render all items and save to hdf5 file
            with h5py.File(h5path, 'r+', libver='latest') as hf:
                loop = tqdm(
                    it, initial=initial, total=num_items)
                for idx, item in loop:
                    name = f'n{idx:0{num_digits}d}'
                    azim, elev, theta, cad_idx = item

                    # append data of item to dictionary
                    annot_dict['name'].append(name)
                    annot_dict['azimuth'].append(azim)
                    annot_dict['elevation'].append(elev)
                    annot_dict['theta'].append(theta)
                    annot_dict['distance'].append(
                        CATEGORY_DISTANCES[self._object_category])
                    annot_dict['cad_index'].append(cad_idx)

                    # continue to next image
                    if idx < start:
                        loop.set_description('Skipping existing')
                        # time.sleep(0.01)
                        continue
                    else:
                        loop.set_description('Generating')

                    # if image exists, continue to next one
                    if not self._data_from_scratch:
                        cat_group = hf['/renderings_db/' + self._object_category]
                        for rt in ['rendering', 'silhouette', 'depth', 'normals']:
                            exists = (name + '_' + rt) in cat_group.keys()
                            if self._positive_type == rt and exists:
                                continue
                            if self._positive_type == 'all' and not exists:
                                break

                    # generate renderings and preprocess them for storing
                    cad_model = self.get_cad_model(cad_idx)
                    rendering, silhouette, depth, normals = \
                        self.generate_rendering(
                            cad_model, azim - radians(AZIMUTH_OFFSET), elev,
                            theta, None)
                    if self._positive_type in ['rendering', 'all']:
                        rendering = cv2.normalize(
                            rendering, None, 0, 255, cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
                    if self._positive_type in ['silhouette', 'all']:
                        silhouette = cv2.normalize(
                            silhouette, None, 0, 255, cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
                    if self._positive_type in ['depth', 'all']:
                        depth = cv2.normalize(
                            depth, None, 0, 1, cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F)
                    if self._positive_type in ['normals', 'all']:
                        normals = cv2.normalize(
                            normals, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # save in hdf5 format
                    cat_group = hf['/renderings_db/' + self._object_category]
                    for img, type_, ft in zip(
                            [rendering, silhouette, depth, normals],
                            ['rendering', 'silhouette', 'depth', 'normals'],
                            ['.jpg', '.jpg', '.tiff', '.tiff']):
                        if self._positive_type not in [type_, 'all']:
                            continue
                        if name + '_' + type_ in cat_group.keys():
                            del cat_group[name + '_' + type_]
                        cat_group.create_dataset(
                            name + '_' + type_, data=np.asarray(cv2.imencode(
                                ft, img)[1].tobytes()))

                    if idx > 0 and idx % 1e5 == 0:
                        initial = idx
                        loop.close()
                        logging.info("Closing and reopening file [%s]" % initial)
                        time.sleep(60)  # wait 60 seconds
                        break
                else:
                    break  # break out of while loop when inner loop is done

        # convert dictionary to pandas dataframe and save to file
        save_path = os.path.join(
            self._root_dir, self._data_dir, 'annotations_csv',
            self._object_category + '_database_annotations.csv')
        # if not os.path.exists(save_path) or self._data_from_scratch:
        df = pd.DataFrame.from_dict(annot_dict)
        df.to_csv(save_path, index=False)

    def plot_distributions(self):
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        sns.kdeplot(data=(self._annot_df['azimuth']-pi)/pi*180, ax=ax[0])
        sns.kdeplot(data=(self._annot_df['elevation'])/pi*180, ax=ax[1])
        sns.kdeplot(data=(self._annot_df['theta'])/pi*180, ax=ax[2])
        for x in ax:
            x.axvline(x=0, color='red', linestyle='--')
        return fig