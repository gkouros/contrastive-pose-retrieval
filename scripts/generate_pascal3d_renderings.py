#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import pandas as pd
import logging
import torch
from absl import flags

sys.path.append(sys.path[0] + '/../src')
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset
from dataloader.pascal3dplus_constants import OBJECT_CATEGORIES

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('from_scratch', False, 'Set to true to render even if '
                     'the renderings already exist')
flags.DEFINE_enum('object_category', 'car', OBJECT_CATEGORIES + ['all'],
                  'The object category to render for')
flags.DEFINE_enum('positive_type', 'all',
                  ['rendering', 'silhouette', 'depth', 'normals', 'all'],
                  'The type of rendering to generate and save')
flags.DEFINE_string('root_dir',
                    '/esat/topaz/gkouros/datasets/pascal3d',
                    'The path to the dataset')
flags.DEFINE_integer('downsample_rate', 2, 'Rendering downsampling rate.'
                     'Increase for efficiency against quality')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'test', 'all', 'db'],
                  'Selects which parts of the data to render for')
flags.DEFINE_boolean('use_fixed_cad_model', False, 'If true use the 1st cad '
                     'model for all renderings')
flags.DEFINE_string('db_name', 'db', 'The name of the extended database')

if __name__ == '__main__':
    # initialize command line arguments
    FLAGS(sys.argv)
    # Set the cuda device
    if not torch.cuda.is_available():
        raise Exception('Cuda not enabled')
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    logging.info('device: %s' % device)

    if FLAGS.object_category == 'all':
        categories = [x for x in OBJECT_CATEGORIES if 'FGL' not in x]
    else:
        categories = [FLAGS.object_category]

    if FLAGS.split in ['train', 'val', 'test', 'all']:
        if FLAGS.split == 'all':
            sets = ['train', 'test']
        else:
            sets = [FLAGS.split]

        for split in sets:
            for category in categories:
                # create dataset
                dataset = PASCAL3DPlusDataset(
                    root_dir=FLAGS.root_dir,
                    split=split,
                    object_category=category,
                    transforms=None,
                    subset_ratio=1.0,
                    positive_type=FLAGS.positive_type,
                    device=device,
                    data_from_scratch=False,
                    render_from_scratch=FLAGS.from_scratch,
                    downsample_rate=FLAGS.downsample_rate,
                    labelling_method='quat',
                    use_fixed_cad_model=FLAGS.use_fixed_cad_model,
                    use_hdf5=False,
                )
                logging.info(
                    '%s.%s dataset size: %d' % (split, category, len(dataset)))
                logging.info(
                    'Verifying %s.%s dataset renderings' % (split, category))
                if FLAGS.from_scratch or not dataset.verify_renderings():
                    logging.info('Starting rendering positives for the '
                                 '%s.%s dataset' % (split, category))
                    dataset.render_and_save()
                    logging.info('Rendering finished')
    elif FLAGS.split == 'db':
        for category in categories:
            dataset = PASCAL3DPlusDataset(
                root_dir=FLAGS.root_dir,
                split='train',
                object_category=category,
                transforms=None,
                subset_ratio=1.0,
                positive_type=FLAGS.positive_type,
                device=device,
                data_from_scratch=False,
                render_from_scratch=FLAGS.from_scratch,
                downsample_rate=FLAGS.downsample_rate,
                labelling_method='quat',
                use_fixed_cad_model=FLAGS.use_fixed_cad_model,
                use_hdf5=True,
            )
            logging.info(
                'Verifying %s.%s dataset renderings' % (FLAGS.split, category))
            logging.info('Starting rendering positives for the '
                         '%s.%s dataset' % (FLAGS.split, category))
            dataset.render_extended_db(name=FLAGS.db_name)
            logging.info('Rendering finished')

    else:
        raise ValueError('Invalid split specified')
