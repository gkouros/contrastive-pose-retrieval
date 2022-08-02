import os
import numpy as np
import cv2
import pandas as pd
import torch
from urllib import request
from tqdm import tqdm
import logging
from .dataset_utils import calc_theta_local


class VirtualKITTIv2Dataset(torch.utils.data.Dataset):
    """Virtual KITTI v2 Dataset loader

    Based on samplers.TuplesToWeightsSampler of the pytorch_metric_learning
    library
    """
    def __init__(
            self,
            root_dir,
            scenes=[1],
            categories=["clone", "clone"],
            formats=["rgb", "rgb"],
            transforms=None,
            min_bbox_size=1000,
            occupancy_ratio=0.4,
            train=True,
            subset_ratio=1.0,
            indices_to_keep=None,
            shuffle=False,
            ordered=False,
            download=False,
            segment_instances=False,
            angle_resolution=1,
            random_positive=True,
            horizontal_flip=False,
            ):
        """
        Based on PyTorch Data Loading Tutorial
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        Args:
            root_dir (string): Directory with all the images.
            categories (list/tuple): Dataset categories tuple with values from
                list [real, clone, fog, morning, overcast, rain, sunset]
            formats (list/tuple): Dataset types list with values rgb,
                instanceSegmentation
            transforms (tuple): Optional transforms to be applied
                to anchor and posneg samples.
            min_bbox_size (int): The minimum size to accept for a bbox in pixels
            train (bool): Whether this dataset is intended for training or
                testing/validation
            subset_ratio (float): Percentage of samples to keep
            indices_to_keep (list): List of indices of samples to keep
            shuffle (bool): If true the dataset will be shuffled
            ordered (bool): Determines loading of images in order or not
            download (bool): If true download the kitti dataset
            segment_instances (bool): For segmenting the car instances form bg
            angle_precision (int): The precision of a local angle based label

        Details:
            Example path:
            ../datasets/vkitti2/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
        """
        self.check_pair_info(categories, formats)
        self.root_dir = root_dir
        self.transforms = transforms
        self.segment_instances = segment_instances
        self.resolution = angle_resolution
        self.random_positive = random_positive
        self.horizontal_flip = horizontal_flip
        self.indices_to_keep = indices_to_keep

        # check if dataset is available and download
        if download:
            self.download_dataset(self.root_dir)

        # read and filter metadata
        self.metadata = self.read_metadata(scenes)
        self.filter_metadata(
            min_bbox_size, occupancy_ratio, train, subset_ratio, shuffle)

        # generate labels
        self.labels = self.generate_labels()

        # in case of ordered access create a global index
        self.ordered = ordered
        self.next_idx = 0

    def __len__(self):
        """ Calculates and return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.metadata)

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
        # preprocess provided index
        idx = self.processIndex(idx)
        # logging.info(f'Getting frame with idx={idx}:\n{self.metadata.iloc[idx]}')

        # get anchor, positive pair and label
        anchor, positive = self.get_instance_pair(idx)
        label = self.labels[idx]

        if self.random_positive:
            # search for another sample with the same label
            positive_indices = np.nonzero(self.labels == label)[0]
            if len(positive_indices) > 0:
                idx2 = np.random.choice(positive_indices)
                _, positive = self.get_instance_pair(idx2)


        # do horizontal flip with a 50% probability
        if self.horizontal_flip:
            prob = np.random.rand()
            if prob > 0.5:
                anchor = anchor[:, ::-1]
                positive = positive[:, ::-1]
                new_label = label - 180
                label = new_label if new_label > 0 else 360 - new_label

        # apply transformations to image
        if self.transforms is not None:
            anchor = self.transforms[0](anchor)
            positive = self.transforms[1](positive)

        return anchor, positive, label

    def read_metadata(self, scenes):
        """ Reads the metadata of the objects like bbox, camera/world coords etc

        Args:
            scenes (list): The list of scene ids
        """
        metadata = pd.DataFrame(columns=(
            'scene', 'frame', 'cameraID', 'trackID', 'alpha', 'width', 'height',
            'length', 'world_space_X', 'world_space_Y', 'world_space_Z',
            'rotation_world_space_y', 'rotation_world_space_x',
            'rotation_world_space_z', 'camera_space_X', 'camera_space_Y',
            'camera_space_Z', 'rotation_camera_space_y',
            'rotation_camera_space_x', 'rotation_camera_space_z'))

        for scene in scenes:
            path = os.path.join(
                self.root_dir,
                f'Scene{scene:02}',
                self.categories[0]
            )
            bboxes = pd.read_csv(os.path.join(path, 'bbox.txt'), sep=' ')
            poses = pd.read_csv(os.path.join(path, 'pose.txt'), sep=' ')
            scene_metadata = pd.merge(left=bboxes, right=poses,
                                      left_on=['frame', 'cameraID', 'trackID'],
                                      right_on=['frame', 'cameraID', 'trackID'])
            scene_col = [scene] * len(scene_metadata)
            scene_metadata['scene'] = scene_col
            metadata = metadata.append(scene_metadata)

        return metadata

    def processIndex(self, idx):
        """ Processes given idx, corrects it and returns corresponding frame_id

        Args:
            idx (int): The index to process

        Returns:
            int: The processed index
            int: The frame_id corresponding to the processed idx
        """
        if idx is None and not self.ordered:
            idx = np.random.choice(len(self))
        elif idx is None and self.ordered:
            idx = self.next_idx
            self.next_idx += 1

            if self.next_idx > len(self):
                raise StopIteration
        elif type(idx) == int and (idx < 0 or idx > len(self)):
            raise ValueError(f'Index {idx} is invalid')

        return idx

    def check_pair_info(self, categories, formats):
        """ Checks whether valid categories were provided and stores them

        Args:
            categories (list/tuple): Valid condition categories in dataset
            formats (list/tuple): Valid formats in dataset eg. rgb
        """
        if type(categories) not in (list, tuple):
            raise ValueError('Provided categories')

        available_categories = \
            ['real', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
        available_formats = ['rgb', 'instanceSegmentation', 'depth']

        for c in categories:
            if c not in available_categories:
                raise ValueError(f'Category {c} is invalid.'
                                 f'Choose from {available_categories}.')

        self.categories = categories

        for f in formats:
            if f not in available_formats:
                raise ValueError(f'Format {c} is invalid.'
                                 f'Choose from {available_formats}.')
        self.formats = formats

    def generate_labels(self):
        """ Generates artificial labels for the dataset

            Details:
                Every pair gets a unique label equal to their index in the
                processed  metadata

            Returns:
                np.array: The vector of labels of the dataset
        """
        # labels = np.arange(len(self))
        # return labels
        # labels = range(-180 * 10**self.precision, 180 * 10**self.precision)
        # labels = range(360 * 10**self.precision)
        labels = [self.get_label(idx) for idx in range(len(self.metadata))]
        return np.array(labels)

    def get_label(self, idx):
        """ Returns the local angle based label of the sample

        Args:
            idx (int): The idx of the sample for extracting its metadata

        Returns:
            int: The local angle based label of the sample
        """
        angle = 180 + calc_theta_local(self.metadata.iloc[idx], to_degrees=True)
        bins = np.linspace(0, 360, 360//self.resolution+1)
        label = np.digitize(angle - self.resolution, bins)
        return label

    def filter_metadata(self,
                        min_bbox_size=0,
                        occupancy_ratio=0,
                        train=True,
                        subset_ratio=1.0,
                        shuffle=False):
        """ Filters out small objects per frame and resulting empty frames

        Args:
            min_bbox_size (int): Minimum number of pixels perf valid bbox
            train (bool): determines whether to keep the first or last part of
                the dataset
            subset_ratio (float): The percentage of samples to keep
            shuffle (bool): whether to shuffle the remaining valid indices
        """
        # ignore instances with small objects
        self.metadata = self.metadata[
            (self.metadata.number_pixels > min_bbox_size) &
            (self.metadata.occupancy_ratio > occupancy_ratio)
        ]

        # ignore wrongly annotated instances with bounding box (0,0,0,0)
        indices_to_remove = self.metadata[
                (self.metadata.left == 0) &
                (self.metadata.right == 0) &
                (self.metadata.top == 0) &
                (self.metadata.bottom == 0)].index
        self.metadata.drop(indices_to_remove, inplace=True)

        # keep only specified indices when using stratified split
        if self.indices_to_keep is not None:
            if not isinstance(self.indices_to_keep, list):
                raise ValueError('indices_to_keep must be a list')
            else:
                self.metadata = self.metadata.iloc[self.indices_to_keep]

        # shuffle
        if shuffle:
            self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)

        # the subset is taken from the beginning if train is true else the end
        new_len = int(subset_ratio * len(self))
        if train:
            self.metadata = self.metadata[:new_len]
        else:
            self.metadata = self.metadata[-new_len:]

    def get_instance_pair(self, idx):
        """ Loads and returns a frame

        Args:
            idx (int): Index of frame in filtered metadata to be used as label
            frame_id (int): ID of corresponding frame

        Returns:
            np.array: The anchor image
            np.array: The positve image

        Details:
            Should only be called in pair mode.
        """
        # extract ids of scene, frame, and camera
        sid = self.metadata.iloc[idx].scene
        fid = self.metadata.iloc[idx].frame
        cid = self.metadata.iloc[idx].cameraID

        # load anchor and positive frames
        anchor, positive = self.load_frame_pair(sid, fid, cid)

        # get instance segmentation mask
        mask = self.load_frame_from_file(
            'clone', 'instanceSegmentation', sid, fid, cid)

        # get bounding box of instance
        bbox = self.metadata.iloc[idx][['left', 'right', 'top', 'bottom']]

        # crop instances from frames
        f1, f2 = self.formats
        anchor = self.crop(anchor, bbox, f1, mask=mask)
        positive = self.crop(positive, bbox, f2, mask=mask)

        return anchor, positive

    def crop(self, frame, bbox, format_, mask=None):
        """ Returns the cropped object instance

        Args:
            frame (np.array): The frame to extract the crop from
            bboxes (pd.DataFrame): The dataframe containing the bounding boxes
            format_ (str): The format of the image eg. rgb/instanceSegmentation
            mask (np.array): None or instance mask to segment anchor instance
        """
        # extract upper left lower right corners of bounding box
        x1, y1 = int(bbox['left']), int(bbox['top'])
        x2, y2 = int(bbox['right']), int(bbox['bottom'])

        # crop image
        img = frame[y1:y2, x1:x2]

        # if target format is instance segmentation, filter out other instances
        if format_ == 'instanceSegmentation':
            img = self.extract_largest_instance_mask(img)
        elif format_ == 'depth':
            img = cv2.normalize(img, None, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

        # segment main instance in crop if flag is set
        if self.segment_instances and format_ != 'instanceSegmentation':
            if mask is None:
                raise ValueError('segment_instances is true but mask is null')
            mask = mask[y1:y2, x1:x2]
            mask = self.extract_largest_instance_mask(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = cv2.bitwise_and(img, img, mask=mask)

        return img

    def load_frame_pair(self, scene_id, frame_id, camera_id):
        """ Returns the anchor and positive images corresponding to frame_id

        Args:
            scene_id (int): The id of the scene
            frame_id (int): The id of the frames to return

        Return:
            tuple: The tuple of anchor and positive images
        """
        anchor = self.load_frame_from_file(
            self.categories[0], self.formats[0], scene_id, frame_id, camera_id)
        positive = self.load_frame_from_file(
            self.categories[1], self.formats[1], scene_id, frame_id, camera_id)

        return anchor, positive

    def load_frame_from_file(self, category, format_,
                             scene_id, frame_id, camera_id):
        """ Load a frame

        Args:
            category (string): The category type specifying subdirectory1
            format_ (string): The format of the frame specifying subdirectory2
            scene_id (int): The id of the scene
            frame_id (int): The id of the frame
            camera_id (int): The id of the camera (0 or 1)

        Returns:
            np.array: The frame that was read
        """
        format_map = {
            'rgb': {'prefix': 'rgb_', 'type': 'jpg'},
            'instanceSegmentation': {'prefix': 'instancegt_', 'type': 'png'},
            'depth': {'prefix': 'depth_', 'type': 'png'}
        }
        prefix = format_map[format_]['prefix'] if category != 'real' else '0'
        filetype = format_map[format_]['type'] if category != 'real' else 'png'
        path = os.path.join(
                self.root_dir,
                f'Scene{scene_id:02}',
                category,
                'frames',
                format_,
                f'Camera_{camera_id}',
                f'{prefix}{frame_id:05}.{filetype}'
            )
        frame = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

        return frame

    def extract_largest_instance_mask(self, img):
        """ Returns a mask with the single biggest instance in the frame

        Args:
            img (np.array): The instance segmentatioon image to filter
        """
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        max_num_pixels = 0
        max_color = colors[0]
        for color in colors:
            if all(color == [0, 0, 0]):
                continue
            num_pixels = np.count_nonzero(cv2.inRange(img, color, color))
            if num_pixels > max_num_pixels:
                max_num_pixels = num_pixels
                max_color = color

        #  get mask of max color instance
        mask = cv2.inRange(img, max_color, max_color)

        # convert mask to color image
        img = np.expand_dims(mask, 2)
        img = np.repeat(img, 3, 2)

        return img

    def download_dataset(self, root_dir):
        """ Download the KITTIv2 dataset
        """
        # raise NotImplementedError
        if os.path.isdir(root_dir):
            logging.info('Dataset is already downloaded')
            return

        # create dataset folder if it doesn't already exist
        try:
            os.mkdir(root_dir)
        except OSError:
            logging.error("Creation of the directory %s failed" % root_dir)
        else:
            logging.info("Successfully created the directory %s " % root_dir)

        logging.info(f'Downloading dataset in {root_dir}')
        root_url = 'http://download.europe.naverlabs.com//virtual_kitti_2.0.3/'

        # download md5 checksum filie
        fn = 'vkitti_2.0.3_md5_checksums.txt'
        download(url=root_url + fn,
                 out=root_dir + '/' + 'vkitti_2.0.3_md5_checksums.txt')

        # download metadata
        fn = 'vkitti_2.0.3_textgt.tar.gz'
        download(url=root_url + fn,
                 out=root_dir + '/' + fn)

        # download datasets
        for format_ in self.formats:
            fn = f'vkitti_2.0.3_{format_}.tar'
            download(url=root_url + fn,
                     out=root_dir + '/' + fn)


def download(url, out):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        request.urlretrieve(url,
                            filename=out,
                            reporthook=t.update_to)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
