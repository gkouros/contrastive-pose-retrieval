#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import pickle
import logging
import torch
from absl import flags
from tqdm.auto import tqdm
import h5py
from torchvision.models import resnet50

sys.path.append(sys.path[0] + '/../src')
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset
from dataloader.pascal3dplus_constants import OBJECT_CATEGORIES, IMAGE_SIZES
from pipeline.datasets import get_pascal3d_train_val_test_datasets
from pipeline.network import create_network, load_weights
from pipeline.testing import MultimodalTwoStreamEmbeddingSpaceTester


torch.backends.cudnn.benchmark = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir',
                    '/esat/topaz/gkouros/datasets/pascal3d',
                    'The path to PASCAL3D')
flags.DEFINE_string('weights_path', '', 'The path to the trained model weights')
flags.DEFINE_string('category', 'car', 'The object category to encode for')

if __name__ == '__main__':
    # initialize command line arguments
    FLAGS(sys.argv)
    # Set the cuda device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    logging.info('device: %s' % device)
    num_workers = 8

    """ get datasets """
    train_dataset, _, _ = \
        get_pascal3d_train_val_test_datasets(
            root_dir=FLAGS.root_dir,
            subset_ratio=1.0,
            positive_type='normals',
            positive_from_db=False,
            use_fixed_cad_model=False,
            device=torch.device('cpu'),
            labelling_method='euler',
            object_category=FLAGS.category,
            object_subcategory=0,
            downsample_rate=2,
            use_hdf5=False,
            bbox_noise=0,
            evaluation=True,
            to_bgr=False,
        )

    models = create_network(
        device=device,
        backbone=resnet50,
        embedding_size=512,
        multimodal=True,
        pretrained=True,
    )
    trunk = models['trunk']
    embedder = models['embedder']
    load_weights(FLAGS.weights_path, trunk, embedder, multimodal=True, device=device)

    # set model to inference mode
    trunk.eval()
    embedder.eval()
    logging.info('Model has been created!')

    """ Create the tester """
    Tester = MultimodalTwoStreamEmbeddingSpaceTester
    tester = Tester(
        normalize_embeddings=True,
        use_trunk_output=False,
        batch_size=128,
        dataloader_num_workers=num_workers,
        pca=None,
        data_device=device,
        dtype=None,
        data_and_label_getter=None,
        label_hierarchy_level='all',
        dataset_labels=train_dataset.labels,
        set_min_label_to_zero=False,
        accuracy_calculator=None,
        visualizer=None,
        visualizer_hook=None,
        end_of_testing_hook=None,
    )

    # compute training embeddings for reference set
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, num_workers=num_workers)
    fp = os.path.join(FLAGS.weights_path, 'pascal3d_trainset_embeddings.pkl')
    anchors, posnegs, labels = tester.compute_all_embeddings(
        train_dataloader, trunk, embedder, split='train')
    if tester.normalize_embeddings:
        anchors = torch.nn.functional.normalize(anchors)
        posnegs = torch.nn.functional.normalize(posnegs)
    with open(fp, 'wb') as f:
        pickle.dump((anchors, posnegs, labels), f)

    logging.info('Encoding completed successfully')
    logging.info(f'Number of ref set embeddings: {len(anchors)}')