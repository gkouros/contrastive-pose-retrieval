import os
import numpy as np
import cv2
import pandas as pd
import json
import torch
from tqdm.auto import tqdm
import logging
from collections import OrderedDict
from matplotlib import pyplot as plt
import seaborn as sns
from math import atan2, degrees
from angles import normalize
from copy import deepcopy

# local imports
from dataloader.renderer import Renderer
import dataloader.apolloscape_car_models as car_models
import dataloader.apolloscape_utils as uts

# frame dimensions
HEIGHT = 2710
WIDTH = 3384


class ApolloScapeDataset(torch.utils.data.Dataset):
    """ Dataset class for the ApolloScape 3D car understanding dataset"""

    def __init__(
            self,
            root_dir,
            split='train',
            transforms=None,
            min_bbox_size=5000,
            min_occupancy_ratio=0.5,
            subset_ratio=1.0,
            shuffle=False,
            angle_resolution=1,
            random_positive=True,
            horizontal_flip=True,
            use_blacklist=True,
            bbox_expansion_factor=0.25,
            positive_type='rendering',
            render_scale=0.2,
            load_metadata=False,
            device=torch.device('cpu')
            ):
        """
        Based on PyTorch Data Loading Tutorial
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            root_dir (string): Directory with all the images.
            split (str): train or val split
            transforms (tuple): Optional transforms to be applied
                to anchor and posneg samples.
            min_bbox_size (int): The minimum size to accept for a bbox
            min_occupancy_ratio (float): The minimum occupancy ratio for
                a valid instance
            subset_ratio (float): The ratio of data to use
            shuffle (bool): Flag for shuffling the data
            angle_resolution (float): The resolution of angle labels
            random_positive (bool): Selecting a random positive for an anchor
            horizontal_flip (bool): Add hflip for data augmentation
            use_blacklist (bool): For filtering manually selected images
            bbox_expansion_factor (float): Factor for enlarging the bboxes
            positive_type (str): Determines the format of positives
            render_scale (float): Rescaling factor of renderings to fit in GPU
            load_metadata (bool): Load metadata instead of creating again
            device (torch.device): Device to use for rendering

        Details:
            Example path:
            ../datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
        """
        self._root_dir = root_dir
        self._split = split
        self._transforms = transforms
        self._resolution = angle_resolution
        self._random_positive = random_positive
        self._horizontal_flip = horizontal_flip
        self._min_bbox_size = min_bbox_size
        self._min_occupancy_ratio = min_occupancy_ratio
        self._bbox_expansion_factor = bbox_expansion_factor
        self._pos_type = positive_type
        self._render_scale = render_scale
        self._datan_config = {}
        self._poses = {}
        self._frames = []
        self._models = {}
        self._metadata = None
        self._shuffle = shuffle
        self._idx = 0
        self._num_classes = 360 // self._resolution
        self._subset_ratio = subset_ratio
        self._load_metadata = load_metadata
        self._device = device

        # prepare dataset
        logging.info(f'Preparing {split} dataset')
        self.get_data_parameters()
        self.get_3d_car_understanding_config()
        self._frames = self.load_frames()
        self._metadata = self.construct_dataframe()
        if use_blacklist:
            self.remove_blacklisted_instances()
        self.labels, self.sample_weights = self.generate_labels()
        logging.info(f'{self._split} dataset initialized')

    def __len__(self):
        """ Calculates and return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self._metadata)

    def __getitem__(self, idx=None):
        """ Returns either a random item or an indexed item

        Args:
            frame_idx (int/None): If int, return indexed item, else random item

        Returns if self.return_pairs == True:
            np.array: The anchor image
            np.array: The positive image
            int: The label of the anchor and/or positive images

        Returns if self.return_pairs == False:
            np.array: The image based on the provided index or a random one
            int: The label of the image

        Details:
            If self.return_pair is true, then returns an anchor, positive pair
            along with the corresponding label. Otherwise, returns imaage based
            on object.
        """
        # preprocess idx
        idx = idx if idx is not None else (self._idx + 1) % len(self)
        self._idx = idx
        # get anchor, positive pair and label
        anchor, positive, label = self.get_sample_pair(idx, self._pos_type)

        if self._random_positive:
            # search for another sample with the same label
            positive_indices = np.nonzero(self.labels == label)[0]
            if len(positive_indices) > 0:
                idx2 = np.random.choice(positive_indices)
                _, positive, _ = self.get_sample_pair(idx2, self._pos_type)

        # apply transformations to image
        if self._transforms is not None:
            anchor = self._transforms[0](anchor)
            positive = self._transforms[1](positive)

        return anchor, positive, label

    def generate_labels(self):
        """
        create labels by binning local angles based on the resolution parameter
        """
        self._metadata['label'] = self._metadata['local_angle']
        # bins = np.linspace(0, 360, 360//self._resolution+1)
        # self._metadata['label'] = self._metadata['local_angle'].apply(
        #     lambda x: np.digitize(180 + x - self._resolution, bins)
        # )
        self._metadata['label'] = self._metadata['local_angle'].apply(
            lambda x: int(round((x + 180) / self._resolution)
                          % (360 // self._resolution))
        )
        labels = self._metadata['label'].values
        classes = np.arange(0, self._num_classes+1, dtype=np.uint16)
        class_weights = np.zeros_like(classes).astype(np.float32)
        for c in classes:
            try:
                class_weights[c] = 1.0 / len(np.where(labels == c)[0])
            except ZeroDivisionError:
                pass
        sample_weights = np.array([class_weights[label] for label in labels])
        return labels, sample_weights

    def create_renderer(self, intrinsics=None):
        h = int(HEIGHT * self._render_scale)
        w = int(WIDTH * self._render_scale)
        return Renderer(self._device, image_size=(h, w), intrinsics=intrinsics)

    def get_data_parameters(self):
        """get the data configuration of the dataset.
        These parameters are shared across different tasks
        """
        self._data_config = {}
        self._data_config['image_size'] = [HEIGHT, WIDTH]

        # fx, fy, cx, cy
        self._data_config['intrinsics'] = {
            'Camera_5': np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array(
                [2300.39065314361, 2301.31478860597,
                 1713.21615190657, 1342.91100799715])}

        # normalized intrinsic for handling image resizing
        cam_names = self._data_config['intrinsics'].keys()
        for c_name in cam_names:
            self._data_config['intrinsics'][c_name][[0, 2]] /= \
                self._data_config['image_size'][1]
            self._data_config['intrinsics'][c_name][[1, 3]] /= \
                self._data_config['image_size'][0]

    def get_3d_car_understanding_config(self):
        """ get configuration of the dataset for 3d car understanding
        """
        ROOT = self._root_dir
        self._data_config['image_dir'] = \
            os.path.join(ROOT, 'train', 'images')
        self._data_config['pose_dir'] = \
            os.path.join(ROOT, 'train', 'car_poses')
        self._data_config['train_list_filepath'] = \
            os.path.join(ROOT, 'train', 'split', 'train-list.txt')
        self._data_config['val_list_filepath'] = \
            os.path.join(ROOT, 'train', 'split', 'validation-list.txt')
        self._data_config['test_list_filepath'] = \
            os.path.join(ROOT, 'train', 'split', 'test-list.txt')
        self._data_config['car_model_dir'] = \
            os.path.join(ROOT, 'train', 'car_models')
        self._data_config['car_instances_dir'] = \
            os.path.join(ROOT, 'train', 'car_instances')
        self._data_config['car_renderings_dir'] = \
            os.path.join(ROOT, 'train', 'car_renderings')
        self._data_config['car_blacklists_dir'] = \
            os.path.join(ROOT, 'train', 'car_blacklists')

        # create instance and renderings directories if they don't exist
        for dirname in ['car_instances_dir', 'car_renderings_dir']:
            dirpath = self._data_config[dirname]
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

    def get_intrinsics(self, image_name, camera_name=None, mat=False):
        assert self._data_config
        K = None
        if camera_name:
            K = self._data_config['intrinsics'][camera_name]
        elif 'Camera' in image_name:
            camera_name = 'Camera_' + image_name[-1]
            K = self._data_config['intrinsics'][camera_name]
        else:
            for name in self._data_config['intrinsics'].keys():
                if name in image_name:
                    K = self._data_config['intrinsics'][name]
        if mat:
            K = uts.intrinsic_vec_to_mat(K, self._data_config['image_size'])
        return K

    def load_car_models(self):
        """Load all the car models
        """
        models = OrderedDict([])
        model_dir = self._data_config['car_model_dir']
        for model in tqdm(car_models.models, desc='Loading car models'):
            car_model = '%s/%s.json' % (model_dir, model.name)
            with open(car_model, 'rb') as f:
                models[model.name] = json.load(f)
                models[model.name]['vertices'] = np.array(
                    models[model.name]['vertices'])
                models[model.name]['faces'] = np.array(
                    models[model.name]['faces']) - 1  # add index offset
        return models

    def load_car_poses(self):
        root = self._data_config['pose_dir']
        car_poses = {}
        for name in tqdm(self._frames, desc='Loading car poses'):
            filepath = os.path.join(root, name + '.json')
            with open(filepath, 'r') as f:
                car_poses[name] = json.load(f)
        return car_poses

    def load_frames(self):
        path = self._data_config[self._split + '_list_filepath']
        # read image names from file based on split (train or val)
        with open(path, 'r') as f:
            _frames = f.readlines()
        # remove newline character and filetype suffix
        _frames = [name.strip()[:-4] for name in _frames]
        return sorted(_frames)

    def get_model(self, id):
        # load models if not already loaded
        if len(self._models) == 0:
            self._models = self.load_car_models()

        return self._models[car_models.car_id2name[id].name]

    def get_frame(self, name, to_rgb=False):
        path = os.path.join(self._root_dir, 'train', 'images')
        img_path = os.path.join(path, name + '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def remove_blacklisted_instances(self):
        path = self._data_config['car_blacklists_dir']
        fn = f'{path}/{self._split}set_blacklist.csv'
        if not os.path.exists(fn):
            logging.info(f'Blacklist for {self._split}set not found. '
                         'Keeping all instances.')
            return
        # read blacklist file
        blacklist = list(pd.read_csv(fn, sep=',', header=None).values.ravel())
        # append horizontally flipped instances to be removed as well
        blacklist = blacklist + [x + 1 for x in blacklist]
        # remove blacklisted instances
        self._metadata.drop(blacklist, inplace=True)
        self._metadata.reset_index(drop=True, inplace=True)

    def construct_dataframe(self):
        path = self._data_config['car_instances_dir']
        fn = f'{path}/{self._split}.csv'
        if os.path.exists(fn) and self._load_metadata:
            logging.info(f'Loading annotation file for {self._split} data')
            df = pd.read_csv(fn, index_col=0)
        else:
            logging.info(f'Creating annotation file for {self._split} data')
            self._models = self.load_car_models()
            self._poses = self.load_car_poses()
            df = pd.DataFrame(columns=[
                'frame', 'camera_id', 'car_model_id', 'area', 'visible_rate',
                'x', 'y', 'z', 'yaw', 'pitch', 'roll',  # pose
                'left', 'right', 'top', 'bottom',  # bounding box
                'global_angle', 'ray_angle', 'local_angle'  # rotation angles
            ])
            instances = []

            for frame in tqdm(self._frames):
                instance = {}
                instance['frame'] = frame
                instance['camera_id'] = frame[-1]
                for car_idx, pose in enumerate(self._poses[frame]):
                    if pose['area'] < self._min_bbox_size or \
                            pose['visible_rate'] < self._min_occupancy_ratio:
                        continue
                    instance['car_model_id'] = pose['car_id']
                    instance['car_idx'] = car_idx
                    instance['area'] = pose['area']
                    instance['visible_rate'] = pose['visible_rate']
                    instance['roll'] = pose['pose'][0]
                    instance['pitch'] = pose['pose'][1]
                    instance['yaw'] = pose['pose'][2]
                    instance['x'] = pose['pose'][3]
                    instance['y'] = pose['pose'][4]
                    instance['z'] = pose['pose'][5]
                    bbox = self.get_bbox(frame, pose)
                    instance['left'] = bbox[0]
                    instance['right'] = bbox[1]
                    instance['top'] = bbox[2]
                    instance['bottom'] = bbox[3]
                    instance['horizontal_flip'] = False
                    grl = self.calculate_angles(pose, offset=-90)
                    instance['global_angle'] = grl[0]
                    instance['ray_angle'] = grl[1]
                    instance['local_angle'] = grl[2]
                    instances.append(deepcopy(instance))

                    # append a horizontally flipped version of the same image
                    instance['horizontal_flip'] = True
                    instance['ray_angle'] = -grl[1]
                    instance['global_angle'] = normalize(
                        180 - grl[0], -180, 180)
                    instance['local_angle'] = normalize(
                        180 - grl[2], -180, 180)
                    instances.append(deepcopy(instance))

            # add instances to dataframe and save to file
            df = pd.DataFrame(instances)
            df.to_csv(fn)

        # shuffle dataframe if flag is set
        df = df.sample(frac=1).reset_index(drop=True) if self._shuffle else df
        # remove horizontally flipped
        df = df.loc[~df['horizontal_flip'] | self._horizontal_flip]
        # keep only a subset of the instances
        df = df.head(int(len(df) * self._subset_ratio))

        return df

    def calculate_angles(self, pose_dict, offset=0):
        roll, pitch, yaw, x, y, z = pose_dict['pose']
        # calculate ray angle
        theta_ray = degrees(atan2(x, z))
        # calculate global angle
        rot_mat = uts.euler_angles_to_rotation_matrix([roll, pitch, yaw])
        vec1 = [0, 0, 1]  # the original z-axis
        vec2 = rot_mat[:, 2]  # the transformed z-axis
        x1, _, z1 = vec1
        x2, _, z2 = vec2
        theta_global = degrees(atan2(x2, z2))
        # calculate local angle
        theta_local = theta_global - theta_ray + offset
        theta_local = normalize(theta_local, -180, 180)
        return theta_global, theta_ray, theta_local

    def render_wireframe_car(self, frame, data):
        model = deepcopy(self.get_model(data['car_model_id']))
        model['vertices'][:, [0, 1]] *= -1
        intrinsics = self.get_intrinsics(frame, frame[-8:])
        intrinsics = uts.intrinsic_vec_to_mat(intrinsics,
                                              self._data_config['image_size'])
        pose = np.array(data[['roll', 'pitch', 'yaw', 'x', 'y', 'z']])

        # project 3D points to 2d image plane
        rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
        rvect, _ = cv2.Rodrigues(rmat)
        tvect = pose[3:].astype(float)
        imgpts, jac = cv2.projectPoints(
            np.float32(model['vertices']), rvect, tvect, intrinsics,
            distCoeffs=None)
        mask = np.zeros(self._data_config['image_size'] + [3])
        for face in np.array(model['faces']) - 1:
            pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]]
                            for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], True, (0, 255, 0), thickness=1)

        return mask

    def get_bbox(self, frame, data):
        # render mask
        cam = frame[-8:]
        intrinsics = self.get_intrinsics(frame, cam)
        intrinsics = uts.intrinsic_vec_to_mat(intrinsics,
                                              self._data_config['image_size'])
        model = deepcopy(self.get_model(data['car_id']))
        model['vertices'][:, [0, 1]] *= -1
        pose = np.array(data['pose'])
        # project 3D points to 2d image plane
        rmat = uts.euler_angles_to_rotation_matrix(pose[:3])
        rvect, _ = cv2.Rodrigues(rmat)
        imgpts, _ = cv2.projectPoints(np.float32(model['vertices']), rvect,
                                      pose[3:], intrinsics, distCoeffs=None)
        imgpts = imgpts.astype(np.int32)
        # calculate bounding box
        left, right = min(imgpts[:, 0, 0]), max(imgpts[:, 0, 0])
        top, bottom = min(imgpts[:, 0, 1]), max(imgpts[:, 0, 1])

        # clip bounding box
        left = np.clip(left, 0, self._data_config['image_size'][1])
        right = np.clip(right, 0, self._data_config['image_size'][1])
        top = np.clip(top, 0, self._data_config['image_size'][0])
        bottom = np.clip(bottom, 0, self._data_config['image_size'][0])
        left, right = round(left), round(right),
        top, bottom = round(top), round(bottom)
        return (left, right, top, bottom)

    def get_annotated_frame(self, idx, wireframe=True):
        frame = self._frames[idx]
        img = self.get_frame(frame)
        instances = self._metadata[
            (self._metadata.frame == frame) &
            (not self._metadata.horizontal_flip)
        ]
        masks = []
        for car_idx in range(len(instances)):
            mask = self.render_wireframe_car(frame, instances.iloc[car_idx])
            masks.append(mask)
            instance = instances.iloc[car_idx]
            pt1 = (instance['left'], instance['top'])
            pt2 = (instance['right'], instance['bottom'])
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)
        if wireframe:
            mask = np.sum(np.array(masks), axis=0)
            img = cv2.addWeighted(img.astype(np.uint8), 1.0,
                                  mask.astype(np.uint8), 0.5, 0)
        return img

    def get_label(self, idx):
        """ Returns the local angle based label of the sample

        Args:
            idx (int): The idx of the car instance for extracting its label

        Returns:
            int: The label based on the local angle of the car instance
        """
        return self._metadata.iloc[idx]['label']

    def plot_ith_sample_pair(self, idx, different_positive=False):
        if different_positive:
            anchor, positive, label = self[idx]
        else:
            anchor, positive, label = self.get_sample_pair(idx)
        self.plot_anchor_positive_pair(anchor, positive, label)
        return self._metadata.iloc[idx]

    def get_sample_pair_at_angle(self, angle, tolerance=1):
        df = self._metadata[((self._metadata['local_angle']
                              - angle).abs() < tolerance)]
        idx = df.sample().index.values[0]
        local_angle = self._metadata.iloc[idx]['local_angle']
        anchor, positive, label = self.get_sample_pair(idx)
        return anchor, positive, label, local_angle

    def get_sample_pair(self, idx, positive_type='rendering'):
        """ Loads and returns a frame

        Args:
            idx (int): Index of frame in filtered metadata to be used as label
            frame_id (int): ID of corresponding frame

        Returns:
            np.array: The anchor image
            np.array: The positve image
        """
        assert positive_type in ['rendering', 'silhouette', 'depth', 'all']

        # read frame
        frame = self._metadata.iloc[idx]['frame']
        img = self.get_frame(frame)

        # get the idx of the car in the image
        car_idx = int(self._metadata.iloc[idx]['car_idx'])

        path = os.path.join(
            self._data_config['car_renderings_dir'],
            f'{frame}_{car_idx}_' + '{}.{}'
        )
        if positive_type not in ['rendering', 'silhouette', 'depth', 'all']:
            raise ValueError('positive_type must be rendering, '
                             'silhouette, depth or all')
        if positive_type in ['rendering', 'all']:
            rendering = cv2.imread(path.format('rendering', 'png'), 0)
            positive = rendering
        if positive_type in ['silhouette', 'all']:
            silhouette = cv2.imread(path.format('silhouette', 'png'), 0)
            positive = silhouette
        if positive_type in ['depth', 'all']:
            depth = cv2.imread(path.format('depth', 'tiff'), -1)
            # remove depth scaling
            depth[depth >= 0] -= depth[depth >= 0].min()
            depth = cv2.normalize(
                depth, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            positive = depth
        if positive_type == 'all':
            # expand dimensions
            rendering = np.expand_dims(rendering, axis=-1)
            silhouette = np.expand_dims(silhouette, axis=-1)
            depth = np.expand_dims(depth, axis=-1)
            positive = np.concatenate((rendering, silhouette, depth), axis=2)

        # resize to the same size as the rgb image
        positive = cv2.resize(positive, img.shape[:2][::-1])

        if len(positive.shape) == 2:
            positive = np.expand_dims(positive, axis=-1)
            positive = np.repeat(positive, 3, axis=-1)

        # crop anchor and positive
        x1, x2, y1, y2 = self._metadata.iloc[idx][
            ['left', 'right', 'top', 'bottom']]
        x1, x2, y1, y2 = self.expand_bbox(x1, x2, y1, y2)
        anchor = img[y1:y2, x1:x2]
        positive = positive[y1:y2, x1:x2]

        # flip horizontally if flag is set
        if self._metadata.iloc[idx]['horizontal_flip']:
            anchor = anchor[:, ::-1]
            positive = positive[:, ::-1]

        label = self._metadata.iloc[idx]['label']

        return (anchor, positive, label)

    def get_sample_pair2(self, frame, car_idx, positive_type='rendering'):
        # get car row from dataframe
        idx = self._metadata.loc[
            (self._metadata.frame == frame) &
            (self._metadata.car_idx == car_idx) &
            (self._metadata.horizontal_flip == False)
        ].index[0]
        return self.get_sample_pair(idx, positive_type)

    def expand_bbox(self, left, right, top, bottom):
        # calculate width and height of bounding box
        h, w = bottom - top, right - left
        # expand bounding box
        left = round(left - w * self._bbox_expansion_factor / 2)
        right = round(right + w * self._bbox_expansion_factor / 2)
        top = round(top - h * self._bbox_expansion_factor / 2)
        bottom = round(bottom + h * self._bbox_expansion_factor / 2)
        # clip bounding box
        left = np.clip(left, 0, self._data_config['image_size'][1])
        right = np.clip(right, 0, self._data_config['image_size'][1])
        top = np.clip(top, 0, self._data_config['image_size'][0])
        bottom = np.clip(bottom, 0, self._data_config['image_size'][0])
        # convert to int
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        return (left, right, top, bottom)

    def render_car_at_pose(self, frame, car_idx):
        # get car row from dataframe
        car = self._metadata.loc[
            (self._metadata.frame == frame) &
            (self._metadata.car_idx == car_idx) &
            (self._metadata.horizontal_flip == False)
        ].iloc[0]

        # get model
        car_id = car['car_model_id']
        model = deepcopy(self.get_model(car_id))
        model['vertices'][:, [1]] *= -1
        pose = [car['roll'], car['pitch'], car['yaw'],
                car['x'], car['y'], car['z']]

        # create a renderer
        intrinsics = self._data_config['intrinsics']['Camera_' + frame[-1]]
        renderer = self.create_renderer(intrinsics)

        # specify viewpoint
        R = uts.euler_angles_to_rotation_matrix(pose[:3]).T
        T = pose[3:]

        # move to device
        R = torch.tensor(R).unsqueeze(0).to(renderer._device)
        T = torch.tensor(T).unsqueeze(0).to(renderer._device)

        # render car
        silhouette, rendering, depth = renderer.renderRT(model, R, T)

        return silhouette[::-1, ::-1], rendering[::-1, ::-1], depth[::-1, ::-1]
        # return silhouette, rendering, depth

    def render_and_save(self, skip_existing=True):
        if len(self._models) == 0:
            self._models = self.load_car_models()

        for idx in tqdm(range(len(self._metadata))):
            if self._metadata.iloc[idx]['horizontal_flip']:
                """
                no need to render flipped images since we can just flip the
                existing renderings at runtime to save time and space
                """
                continue
            car_idx = int(self._metadata.iloc[idx]['car_idx'])
            frame = self._metadata.iloc[idx]['frame']
            filepath = os.path.join(
                self._data_config['car_renderings_dir'],
                f'{frame}_{car_idx}_' + '{}.{}'
            )

            missing = False
            for type_, format_ in zip(['rendering', 'silhouette', 'depth'],
                                      ['png', 'png', 'tiff']):
                if not os.path.exists(filepath.format(type_, format_)):
                    missing = True
                    break

            if missing:
                rendering, silhouette, depth = \
                    self.render_car_at_pose(frame, car_idx)
                cv2.imwrite(filepath.format('rendering', 'png'), rendering)
                cv2.imwrite(filepath.format('silhouette', 'png'), silhouette)
                cv2.imwrite(filepath.format('depth', 'tiff'), depth)

    def verify_renderings(self):
        not_found = []

        loop = tqdm(range(len(self._metadata)))
        count = 0
        for idx in loop:
            if self._metadata.iloc[idx]['horizontal_flip']:
                continue
            car_idx = int(self._metadata.iloc[idx]['car_idx'])
            frame = self._metadata.iloc[idx]['frame']
            filepath = os.path.join(
                self._data_config['car_renderings_dir'],
                f'{frame}_{car_idx}_' + '{}.{}'
            )

            temp_count = 0
            for type_, format_ in zip(['rendering', 'silhouette', 'depth'],
                                      ['png', 'png', 'tiff']):
                if not os.path.exists(filepath.format(type_, format_)):
                    not_found.append(filepath.format(type_, format_))
                    temp_count = 1

            count += temp_count
            loop.set_description(f'Missing: {count}')

        if len(not_found) > 0:
            logging.warn(f'{count} renderings are missing: \n')
            return False
        else:
            logging.info('Renderings verified succesfully.')
            return True

    def plot_local_angle_and_label_distribution(self):
        # sns.histplot(self.labels-180, bins=360//self._resolution, kde=True)
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        sns.histplot(self._metadata['local_angle'].values, ax=ax[0],
                     bins=360 // self._resolution, kde=True)
        sns.histplot(self.labels, ax=ax[1], bins=360 // self._resolution,
                     kde=True)
        ax[0].set_title('Distribution of local angles in dataset')
        ax[0].set_xlabel('local angle in degrees')
        ax[0].set_ylabel('number of car instances')
        ax[1].set_title('Distribution of labels in dataset')
        ax[1].set_xlabel('label')
        ax[1].set_ylabel('number of car instances')
        plt.show()

    def plot_local_angle_distribution(self, ax=None, name=''):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.histplot(self._metadata['local_angle'].values, ax=ax,
                     bins=360 // self._resolution, kde=True)
        ax.set_title(f'Distribution of local angles in {name} dataset')
        ax.set_xlabel('local angle in degrees')
        ax.set_ylabel('number of car instances')

    def plot_anchor_positive_pair(self, anchor, positive, label='nan'):
        fig, ax = plt.subplots(1, 3, figsize=(30, 20))
        ax[0].imshow(cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB))
        ax[1].imshow(positive)
        positive[positive == 255] = 0
        ax[2].imshow(cv2.cvtColor(cv2.addWeighted(anchor, 1, positive, 0.6, 0),
                                  cv2.COLOR_BGR2RGB))
        ax[0].set_title(f'anchor - label={label}')
        ax[1].set_title(f'positive - label={label}')
        ax[2].set_title(f'blended - label={label}')

    def plot_all_sample_pairs_at_angle(self, angle, tolerance=1, max_len=5,
                                       positive_type='rendering'):
        dthetas = (self._metadata['local_angle'] - angle) % 360
        dthetas = dthetas - 360.0 * (dthetas > 180.0)
        in_range = (dthetas.abs() < tolerance)
        not_flipped = (self._metadata['horizontal_flip'] == False)
        df = self._metadata[in_range & not_flipped]
        logging.info(f'Found {len(df)} cars with angle = {angle} and '
                     f'tolerance={tolerance}, but showing {max_len} at most.')
        for idx in df.index[:max_len]:
            try:
                anchor, positive, _ = self.get_sample_pair(idx, positive_type)
                self.plot_anchor_positive_pair(
                    anchor, positive, self._metadata.iloc[idx]['local_angle'])
                plt.show()
            except RuntimeError:
                pass
