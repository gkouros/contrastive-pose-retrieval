#!/usr/bin/env python3
# coding: utf-8

import os
import time
import sys
import numpy as np
import re
import pickle
import logging
import torch
import json
from absl import flags
import gc
import torchvision
from tqdm.auto import tqdm
import h5py
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances.cosine_similarity import CosineSimilarity
from pytorch_metric_learning.distances.lp_distance import LpDistance


sys.path.append(sys.path[0] + '/../src')
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset
from dataloader.pascal3dplus_db_dataset import PASCAL3DPlusDBDataset
from dataloader.pascal3dplus_constants import OBJECT_CATEGORIES, IMAGE_SIZES
from pipeline.datasets import get_pascal3d_train_val_test_datasets
from pipeline.network import create_network
from dataloader.dataset_utils import quaternion_distance
from pipeline.testing import (
    CustomAccuracyCalculator,
    MultimodalTwoStreamEmbeddingSpaceTester,
    GlobalTwoStreamEmbeddingSpaceTester
)


torch.backends.cudnn.benchmark = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_enum('occlusion_level', "0", ["0", "1", "2", "3"], 'The level of occlusion')
flags.DEFINE_string('root_dir',
                    '/esat/topaz/gkouros/datasets/pascal3d',
                    'The path to the dataset')
flags.DEFINE_string('experiment', '', 'The trained model to evaluate')
flags.DEFINE_string('runs_path',
                    '/users/visics/gkouros/projects/models/pascal3d/',
                    'The location of the dataset')
flags.DEFINE_boolean('from_scratch', False, 'Embeddings are computed from scratch')
flags.DEFINE_boolean('evaluate_inference_time', False, 'Estimate inference time')
flags.DEFINE_boolean('evaluate_metrics', True,
                     'Determines if metrics will be calculated')
flags.DEFINE_float('bbox_noise', 0, 'How much noise to add to the bboxes of '
                   'the test set queries')
flags.DEFINE_boolean('use_hdf5', True, 'If true read and write to hdf5 file')
flags.DEFINE_boolean('positive_from_db', False, 'Set to true to load '
                     'renderings from the extended database')
flags.DEFINE_boolean('train_plus_db', False, 'If True use both the training '
                     'and the database renderings')
flags.DEFINE_boolean('bgr', False, 'Whether to evaluate in BGR rather than RGB')

if __name__ == '__main__':
    # initialize command line arguments
    FLAGS(sys.argv)
    # Set the cuda device
    if not torch.cuda.is_available():
        raise Exception('Cuda not enabled')
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    logging.info('device: %s' % device)
    num_workers = 8
    subset_ratio = 1.0
    positive_from_db = FLAGS.positive_from_db
    use_hdf5 = FLAGS.use_hdf5
    runs_path = '/users/visics/gkouros/projects/models/pascal3d/'
    weights_path = f'{FLAGS.runs_path}/{FLAGS.experiment}/saved_models'
    # Create folder for exports
    eval_path = f'{FLAGS.runs_path}/{FLAGS.experiment}/evaluation'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    metrics_to_include = ('accuracy_at_10', 'accuracy_at_30', 'median_error')
    metrics_to_exclude = (
        'recall_at_1', 'recall_at_5', 'nfp', 'precision_at_1', 'r_precision',
        'AMI', 'NMI', 'mean_average_precision', 'mean_average_precision_at_r')

    """ read settings from logs """
    logs_path = f'{FLAGS.runs_path}/{FLAGS.experiment}/logs'
    conf_path = logs_path + '/main.log'
    with open(conf_path, 'r') as f:
        lines = f.read()

    backbone = positive_type = embedding_size = downsample_rate = loss = \
        multimodal = distance = object_category = object_subcategory = \
        batch_size = use_fixed_cad_model = small_db = 0

    for line in lines.split('\n'):
        if 'positive_type' in line:
            positive_type = line.split(':')[1][2:-2]
        if 'embedding_size' in line:
            embedding_size = int(line.split(':')[1][1:-1])
        if 'downsample_rate' in line:
            downsample_rate = int(line.split(':')[1][1:-1])
        if 'multimodal' in line:
            multimodal = line.split(':')[1][1:-1] in ['True', 'true']
        if 'backbone' in line:
            backbone_name = line.split('\':')[1][2:-2]
            if backbone_name == 'resnet18':
                backbone = torchvision.models.resnet18
            else:
                backbone = torchvision.models.resnet50
        if 'loss' in line:
            loss = line.split('\':')[1][2:-2]
        if 'batch_size' in line:
            batch_size = int(line.split(':')[1][1:-1]) * 4
        if 'object_category' in line:
            object_category = line.split('\':')[1][2:-2]
        if 'object_subcategory' in line:
            object_subcategory = int(line.split('\':')[1][1:-1])
        if 'use_fixed_cad_model' in line:
            use_fixed_cad_model = line.split('\':')[1][1:-1] in ['true', 'True']
        if 'small_db' in line:
            small_db = line.split('\':')[1][1:-1] in ['true', 'True']
        if 'warm_start' in line:
            break

    distance = LpDistance() if loss == 'weighted_contrastive' else CosineSimilarity().to(device)
    logging.info('Detected backbone = %s' % backbone_name)
    logging.info('Detected positive type = %s' % positive_type)
    logging.info('Detected embedding size = %s' % embedding_size)
    logging.info('Detected downsample rate = %s' % downsample_rate)
    logging.info('Detected loss = %s' % loss)
    logging.info('Detected multimodal = %s' % multimodal)
    logging.info('Detected distance = %s' % distance)
    logging.info('Detected object_category = %s' % object_category)
    logging.info('Detected object_subcategory = %s' % object_subcategory)
    logging.info('Detected batch_size = %s' % batch_size)
    logging.info('Detected use_fixed_cad_model = %s' % use_fixed_cad_model)
    logging.info('Detected small_db = %s' % small_db)

    if int(FLAGS.occlusion_level) > 0:
        cat_suffix = f'FGL{FLAGS.occlusion_level}_BGL{FLAGS.occlusion_level}'
    else:
        cat_suffix = ''

    """ get datasets """
    train_dataset, _, test_dataset = \
        get_pascal3d_train_val_test_datasets(
            root_dir=FLAGS.root_dir,
            subset_ratio=subset_ratio,
            positive_type=positive_type,
            positive_from_db=False,
            use_fixed_cad_model=use_fixed_cad_model,
            device=torch.device('cpu'),
            labelling_method='quat',
            object_category=object_category + cat_suffix,
            object_subcategory=object_subcategory,
            downsample_rate=downsample_rate,
            use_hdf5=use_hdf5,
            bbox_noise=FLAGS.bbox_noise,
            evaluation=True,
            to_bgr=FLAGS.bgr,
        )
    logging.info(f'Train dataset size: {len(train_dataset)}')
    logging.info(f'Test dataset size: {len(test_dataset)}')

    if FLAGS.positive_from_db:
        db_dataset = PASCAL3DPlusDBDataset(
            root_dir=FLAGS.root_dir,
            object_category=object_category,
            transforms=train_dataset._transforms[1],
            positive_type=positive_type,
            object_subcategory=object_subcategory,
            device=torch.device('cpu'),
            downsample_rate=downsample_rate,
            to_bgr=FLAGS.bgr,
            small_db=small_db,
        )
        logging.info(f'DB dataset size: {len(db_dataset)}')

    models = create_network(
        device=device,
        backbone=backbone,
        embedding_size=embedding_size,
        multimodal=multimodal,
        pretrained=True,
    )
    trunk = models['trunk']
    embedder = models['embedder']
    # load weights
    filenames = os.listdir(weights_path)
    for filename in filenames:
        match1 = re.search('trunk_best', filename)
        match2 = re.search('embedder_best', filename)
        if match1 is not None:
            trunk_weights_fn = filename
        if match2 is not None:
            embedder_weights_fn = filename
        if match1 and match2:
            break

    strict = True
    # load weights of trunk
    trunk_path = os.path.join(weights_path, trunk_weights_fn)
    embedder_path = os.path.join(weights_path, embedder_weights_fn)
    trunk_checkpoint = torch.load(trunk_path, map_location=device)
    embedder_checkpoint = torch.load(embedder_path, map_location=device)
    trunk.module.load_state_dict(trunk_checkpoint, strict=strict)
    embedder.module.load_state_dict(embedder_checkpoint, strict=strict)
    if multimodal:
        trunk.module.anchor_model.module.linear = c_f.Identity()
        trunk.module.posneg_model.module.linear = c_f.Identity()
        embedder.module.anchor_model.module.linear = c_f.Identity()
        embedder.module.posneg_model.module.linear = c_f.Identity()
    else:
        trunk.module.linear = c_f.Identity()
        embedder.module.linear = c_f.Identity()

    # set model to inference mode
    trunk.eval()
    embedder.eval()

    # send model to device
    trunk.to(device)
    embedder.to(device)

    logging.info('Model has been created!')

    """ Create the tester """
    if multimodal:
        Tester = MultimodalTwoStreamEmbeddingSpaceTester
    else:
        Tester = GlobalTwoStreamEmbeddingSpaceTester
    tester = Tester(
        normalize_embeddings=True,
        use_trunk_output=False,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        pca=None,
        data_device=device,
        dtype=None,
        data_and_label_getter=None,
        label_hierarchy_level='all',
        dataset_labels=train_dataset.labels,
        set_min_label_to_zero=False,
        accuracy_calculator=CustomAccuracyCalculator(
            k=1, exclude=metrics_to_exclude, include=metrics_to_include),
        visualizer=None,
        visualizer_hook=None,
        end_of_testing_hook=None,
    )

    # compute training embeddings
    if not FLAGS.positive_from_db or FLAGS.train_plus_db:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers)
        fp = os.path.join(eval_path, 'pascal3d_trainset_embeddings.pkl')
        if not os.path.exists(fp) or FLAGS.from_scratch:
            anchors, posnegs, labels = tester.compute_all_embeddings(
                train_dataloader, trunk, embedder, split='train')
            if tester.normalize_embeddings:
                anchors = torch.nn.functional.normalize(anchors)
                posnegs = torch.nn.functional.normalize(posnegs)
            with open(fp, 'wb') as f:
                pickle.dump((anchors, posnegs, labels), f)
        with open(fp, 'rb') as f:
            embeddings = pickle.load(f)
        train_embeddings = embeddings[1]  # depth / uv maps / render etc. - posnegs
        train_labels = embeddings[2]
        logging.info(f'Number of ref set embeddings: {len(train_embeddings)}')

    # compute testing embeddings
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers)
    fp = os.path.join(
        eval_path, f'pascal3d_testset_L{FLAGS.occlusion_level}_embeddings.pkl')
    if not os.path.exists(fp) or FLAGS.from_scratch:
        anchors, posnegs, labels = tester.compute_all_embeddings(
            test_dataloader, trunk, embedder, split='test')

        if tester.normalize_embeddings:
            anchors = torch.nn.functional.normalize(anchors)
            posnegs = torch.nn.functional.normalize(posnegs)
        with open(fp, 'wb') as f:
            pickle.dump((anchors, posnegs, labels), f)
    with open(fp, 'rb') as f:
        embeddings = pickle.load(f)
    query_embeddings = embeddings[0]  # RGB image - anchors
    query_labels = embeddings[2]
    logging.info(f'Number of query set embeddings: {len(query_embeddings)}')

    # compute database embeddings
    if FLAGS.positive_from_db:
        db_dataloader = torch.utils.data.DataLoader(
            db_dataset, batch_size=batch_size, num_workers=num_workers)
        fp = os.path.join(eval_path, 'pascal3d_dbset_embeddings.pkl')

        if not os.path.exists(fp) or FLAGS.from_scratch:
            embeddings, labels = tester.compute_db_embeddings(
                db_dataloader, trunk, embedder, split='db')
            if tester.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings)
            with open(fp, 'wb') as f:
                pickle.dump((embeddings, labels), f)
        with open(fp, 'rb') as f:
            embeddings = pickle.load(f)
        db_embeddings = embeddings[0]  # depth / uv maps / render etc.-posnegs
        db_labels = embeddings[1]
        logging.info(f'Number of db set embeddings: {len(db_embeddings)}')

    # pick reference set
    if FLAGS.positive_from_db:
        if FLAGS.train_plus_db:
            ref_embeddings = torch.cat((train_embeddings, db_embeddings))
            ref_labels = torch.cat((train_labels, db_labels))
        else:
            ref_embeddings = db_embeddings
            ref_labels = db_labels
    else:
        ref_embeddings = train_embeddings
        ref_labels = train_labels

    try:
        del trunk, embedder, models, trunk_checkpoint, embedder_checkpoint, \
            train_dataloader, test_dataloader, anchors, posnegs, labels
    except Exception:
        pass

    torch.cuda.empty_cache()
    gc.collect()

    if FLAGS.evaluate_metrics:
        if not FLAGS.positive_from_db:
            # calculate metrics
            metrics = tester.accuracy_calculator.get_accuracy(
                query_embeddings, ref_embeddings,
                query_labels, ref_labels,
                embeddings_come_from_same_source=False,
                include=metrics_to_include,
                exclude=metrics_to_exclude,
            )
        else:
            distances = np.zeros(len(query_embeddings))
            for idx in tqdm(range(len(query_embeddings)), desc='Query NN'):
                # for idx in tqdm(range(20), desc='Query NN'):
                query_embedding = query_embeddings[idx].unsqueeze(0)
                query_label = query_labels[idx]
                mat = distance(query_embedding, ref_embeddings)
                _, indices = torch.topk(mat, 1, largest=distance.is_inverted, dim=1)
                min_idx = indices[0].item()
                min_label = ref_labels[min_idx]
                distances[idx] = quaternion_distance(
                    query_label, ref_labels[min_idx], return_degrees=True)

            acc10 = np.count_nonzero(distances < 10) / len(distances)
            acc30 = np.count_nonzero(distances < 30) / len(distances)
            median_error = np.median(distances)
            metrics = {
                'accuracy_at_10': acc10,
                'accuracy_at_30': acc30,
                'median_error': median_error,
            }
    else:
        metrics = {}

    # evaluate inference time
    if FLAGS.evaluate_inference_time:
        # get reference embeddings and labels to compare to
        num_iters = 1 if FLAGS.positive_from_db else 1000
        duration = 0

        for qidx in tqdm(np.random.choice(len(query_embeddings), num_iters),
                         desc='Inference speedtest'):
            start_time = time.time()
            query, _, query_label = test_dataset[qidx]
            query_emb = embedder(trunk(query[None], stream=1), stream=1)
            query_emb = torch.nn.functional.normalize(query_emb)
            query_label = torch.tensor(query_label)[None]

            dist_mat = distance(query_emb, ref_embeddings)
            distances, indices = torch.topk(
                dist_mat, 1, largest=distance.is_inverted, dim=1)
            idx = indices[0].item()

            duration += time.time() - start_time

        metrics['inference_time (ms)'] = (duration / num_iters) * 1000

    metrics['run_id'] = FLAGS.experiment

    logging.info(json.dumps(metrics, indent=4))
