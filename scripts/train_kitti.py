#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import logging
from datetime import datetime
from absl import flags
from math import radians
import shutil

# pytorch imports
import torch
from torch import optim
from torchvision.models import resnet18, resnet50
from pytorch_metric_learning.utils import common_functions as c_f

# local imports
sys.path.append('../src')
sys.path.append('src')
from pipeline.logging_utils import PrettyLog
from pipeline.logging_utils import setup_logging, launch_tensorboard
from pipeline.training import create_trainer
from pipeline.testing import create_tester
from pipeline.metric_learning_utils import create_loss_miner_sampler
from pipeline.optimizers import create_optimizers
from pipeline.network import create_network, load_checkpoint
from pipeline.datasets import get_kitti_train_val_test_datasets
from dataloader.pascal3dplus_constants import OBJECT_CATEGORIES


torch.backends.cudnn.benchmark = True
FLAGS = flags.FLAGS

# script flags
flags.DEFINE_string('checkpoint', '', 'If set to a path, the training '
                    'continues from the last checkpoint')
flags.DEFINE_boolean('logs', False, 'Set to true for collecting logs.')
flags.DEFINE_enum('loglevel', 'info', ['info', 'debug', 'warn', 'error'],
                  'Set logging level.')
flags.DEFINE_string('logdir', './runs', 'Choose path for logs')
flags.DEFINE_boolean('cpu', False, 'Set to true for training on cpu.')
flags.DEFINE_integer('num_workers', 16, 'Set the number of cpu workers')
flags.DEFINE_boolean('tb', False, 'Set to true to launch tensorboard.')
flags.DEFINE_string('experiment', '0', 'The experiment id')

# Dataset Flags
flags.DEFINE_string('dataset_path',
                    '/media/omega/ShubDatasets/KITTI/object',
                    'The path to KITTI object dataset')
flags.DEFINE_string('rendering_path',
                    './NeMo/data',
                    'The path to the rendering dataset')
flags.DEFINE_enum('occlusion_level', 'partly_occluded',
                    ['fully_visible', 'partly_occluded', 'largely_occluded', 'all'],
                    'The occlusion level of the dataset')
flags.DEFINE_float('zmax', 70.0, 'The maximum z value of the dataset')
flags.DEFINE_enum('positive_type', 'normals',
                  ['rendering', 'silhouette', 'depth', 'normals', 'all'],
                  'Selects what type of rendering to use for posnegs.')
flags.DEFINE_boolean('positive_from_db', True, 'Set to true to load '
                     'renderings from the extended database')
flags.DEFINE_float('subset_ratio', 0.9,
                   'Fraction of dataset to use for training. The rest is used '
                   'for validation')
flags.DEFINE_bool('timestamp', False, 'Appends a timestamp to the logdir name')
flags.DEFINE_boolean('horizontal_flip', True,
                     'Enables horizontal flipping with probability=0.5')
flags.DEFINE_enum('object_category', 'car', OBJECT_CATEGORIES,
                  'The object class to train for')
flags.DEFINE_enum('labelling_method', 'quat',
                  ['azim', 'discrete', 'euler', 'quat', 'quat_and_model'],
                  'The method for labelling the samples')
flags.DEFINE_boolean('use_miner', True, 'Training will use a miner if true')
flags.DEFINE_float('pose_positive_threshold', 1, 'The angle in degrees that '
                   'determines if two samples are positive or negative')
flags.DEFINE_boolean('use_fixed_cad_model', False, 'If true the same cad '
                     'model is used for all samples of an object category')
flags.DEFINE_boolean('use_hdf5', False, 'If true read and write to hdf5 file')
flags.DEFINE_integer('object_subcategory', 0, 'If 0 train all object '
                     'subcategories else train all samples with cad model '
                     'index matching the object subcategory')
flags.DEFINE_float('occlusion_scale', 0, 'Determines the level of synthetic '
                   'occlusion. Set to 0 to disable and >0 to enable.')
flags.DEFINE_enum('weight_mode', 'uniform', ['uniform', 'cad'],
                  'Mode for sampling')
flags.DEFINE_float('bbox_noise', 0.0, 'Set to a positive value lower than 1 '
                   'to enable bounding box noise during training')
flags.DEFINE_boolean('augmentations', True, 'Set to False to disable color '
                     'jitter, blurring and horizontal flipping augmenations')
flags.DEFINE_boolean('small_db', False, 'Use 178k db instead of 889k db')
flags.DEFINE_string('kitti_trainval_split_path',
                    'datasets/mv3d_kitti_splits',
                    'The path to KITTI object trainval split')
flags.DEFINE_string('kitti_train_split', 'train.txt',
                    'The path to KITTI object train split')
flags.DEFINE_string('kitti_val_split', 'val.txt',
                    'The path to KITTI object val split')

# Architecture Flags
flags.DEFINE_enum('loss', 'weighted_contrastive',
                  ['contrastive', 'triplet', 'weighted_triplet',
                   'weighted_contrastive', 'margin', 'proxy',
                   'soft', 'arcface', 'dynamic_margin'],
                  'Determines the type of loss function to be used')
flags.DEFINE_boolean('use_nnc_sampler', False, 'Switch from MPerClassSampler '
                     'to NonNeighboringClassesSampler')
flags.DEFINE_integer('embedding_size', 512, 'The size of the final embeddings')
flags.DEFINE_boolean('multimodal', True, 'Dual net for anchors and posnegs')
flags.DEFINE_enum('backbone', 'resnet50', ['resnet18', 'resnet50'],
                  'Select backbone between resnet18 and resnet50')

# optimization params
flags.DEFINE_float('margin', 1.0, 'Margin of triplet loss function')
flags.DEFINE_boolean('use_dynamic_margin', True, 'Set to False to disable '
                     'dynamic margin when using triplet or contrastive loss')
flags.DEFINE_enum('type_of_triplets', 'all', ['all', 'hard', 'semihard'],
                  'Miner selection of hard, semihard, or all triplets')
flags.DEFINE_float('trunk_lr', 0.0001, 'Learning rate for the trunk')
flags.DEFINE_float('trunk_wd', 0.0005, 'Weight decay for the trunk')
flags.DEFINE_float('embedder_lr', 0.001, 'Learning rate for the embedder')
flags.DEFINE_float('embedder_wd', 0.0005, 'Weight decay for the embedder')
flags.DEFINE_enum('lr_scheduler', 'exp', ['none', 'plateau', 'exp', 'step'],
                  'Sets the type of lr scheduler')
flags.DEFINE_integer('lr_decay_patience', 1000,
                     'Epochs in plateau before reducing learning rate')
flags.DEFINE_float('lr_decay', 1.0,
                   'Epochs in plateau before reducing learning rate')

# training params
flags.DEFINE_integer('num_epochs', 100, 'How many epochs to run training for')
flags.DEFINE_integer('num_iters_per_epoch', 0, 'Number of iterations per '
                     'epoch. Set to 0 for max iterations.')
flags.DEFINE_boolean('warm_start', True, 'If true, only embedder is trained '
                     'in the first epoch')
flags.DEFINE_integer('batch_size', 64, 'Specify the batch size for training')
flags.DEFINE_integer('patience', 1000, 'Epoches in plateau before termination')
flags.DEFINE_integer('k', 0, 'k of k-NN for evaluating performance. '
                     'Valid values: 0 -> None, 1 -> max_bin_count, '
                     '>1 -> same value, -1 -> batch_size')
flags.DEFINE_boolean('validate_with_testset', False, 'Set to true to use the '
                     'testset for validation rather than the validation set')
flags.DEFINE_enum('primary_metric', 'accuracy_at_10',
                  ['precision_at_1', 'r_precision', 'mean_average_precision',
                   'mean_average_precision_at_r', 'recall_at_1', 'recall_at_5',
                   'AMI', 'NMI', 'accuracy_at_10', 'accuracy_at_30',
                   'median_error'], 'The primary metric to monitor training')
flags.DEFINE_integer('downsample_rate', 2,
                     'The ratio to downsample image size')

def main():

    # setup logging
    subdir = FLAGS.experiment + FLAGS.timestamp * (
        f'_{datetime.now().strftime("%Y%m%dT%H%M%S")}')
    experiment_dir = os.path.join(FLAGS.logdir, subdir)
    logdir = os.path.join(experiment_dir, 'logs')

    # determine if there is a checkpoint
    checkpoint_dir = os.path.join(FLAGS.logdir, FLAGS.checkpoint)
    exists_checkpoint = (FLAGS.checkpoint != '') \
        and ('saved_models' in os.listdir(checkpoint_dir))
    if exists_checkpoint and checkpoint_dir != experiment_dir:
        shutil.copytree(checkpoint_dir, experiment_dir)

    # setup logging
    setup_logging(save_logs=FLAGS.logs, level=FLAGS.loglevel, logdir=logdir)
    logging.info(PrettyLog({k: FLAGS.get_flag_value(k, 0) for k in FLAGS}))

    # set device
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not FLAGS.cpu else "cpu")

    logging.info(f'Using device={device}')

    # create dataset
    train_dataset, val_dataset, test_dataset = \
        get_kitti_train_val_test_datasets(
            root_dir=FLAGS.dataset_path,
            rendering_dir=FLAGS.rendering_path,
            subset_ratio=FLAGS.subset_ratio,
            zmax=FLAGS.zmax,
            kitti_trainval_split_path=FLAGS.kitti_trainval_split_path,
            kitti_train_split=FLAGS.kitti_train_split,
            kitti_val_split=FLAGS.kitti_val_split,
            positive_type=FLAGS.positive_type,
            positive_from_db=FLAGS.positive_from_db,
            occlusion_level=FLAGS.occlusion_level,
            pose_positive_threshold=FLAGS.pose_positive_threshold,
            use_fixed_cad_model=FLAGS.use_fixed_cad_model,
            object_subcategory=FLAGS.object_subcategory,
            device=torch.device('cpu'),
            labelling_method=FLAGS.labelling_method,
            object_category=FLAGS.object_category,
            downsample_rate=FLAGS.downsample_rate,
            use_hdf5=FLAGS.use_hdf5,
            weight_mode=FLAGS.weight_mode,
            to_bgr=False,
            bbox_noise=FLAGS.bbox_noise,
            augmentations=FLAGS.augmentations,
        )
    if FLAGS.validate_with_testset:
        val_dataset = test_dataset
    logging.info(f'Train dataset size: {len(train_dataset)}, '
                 f'Val dataset size: {len(val_dataset)}')

    # create network
    backbone = eval(FLAGS.backbone)
    models = create_network(
        device=device,
        backbone=backbone,
        embedding_size=FLAGS.embedding_size,
        multimodal=FLAGS.multimodal,
    )
    trunk = models['trunk']
    embedder = models['embedder']
    logging.info(
        f'Feature Embedding: '
        f'Image@3x224x224 -- trunk --> {backbone().fc.in_features} '
        f'-- embedder --> {FLAGS.embedding_size}'
    )

    # Create loss, miner, sampler for deep metric learning
    loss_funcs, mining_funcs, sampler = create_loss_miner_sampler(
        loss_type=FLAGS.loss,
        dataset=train_dataset,
        margin=FLAGS.margin,  # disable margin to use defaults for each loss
        embedding_size=FLAGS.embedding_size,
        batch_size=FLAGS.batch_size,
        type_of_triplets=FLAGS.type_of_triplets,
        device=device,
        use_nnc_sampler=FLAGS.use_nnc_sampler,
        labelling_method=FLAGS.labelling_method,
        use_miner=FLAGS.use_miner,
        positive_pose_threshold=radians(FLAGS.pose_positive_threshold),
        use_dynamic_margin=FLAGS.use_dynamic_margin,
    )

    # create optimizers
    optimizers, lr_schedulers = create_optimizers(
        models=models,
        optimizer=optim.Adam,
        trunk_lr=FLAGS.trunk_lr,
        trunk_wd=FLAGS.trunk_wd,
        embedder_lr=FLAGS.embedder_lr,
        embedder_wd=FLAGS.embedder_wd,
        scheduler=FLAGS.lr_scheduler,
        lr_decay=FLAGS.lr_decay,
        patience=FLAGS.lr_decay_patience,
        multimodal=FLAGS.multimodal,
    )

    # check if there is a loss optimizer and append it to the other optimizers
    if 'metric_loss_optimizer' in loss_funcs:
        optimizers['metric_loss_optimizer'] = \
            loss_funcs['metric_loss_optimizer']
        del loss_funcs['metric_loss_optimizer']

    # set saved model directory
    modeldir = os.path.join(FLAGS.logdir, subdir, 'saved_models')
    tbdir = os.path.join(FLAGS.logdir, subdir, 'tensorboard')

    # create tester for evaluation
    hooks, end_of_epoch_hook = create_tester(
        dataset_dict={"train": train_dataset, "val": val_dataset},
        model_folder=modeldir,
        log_dir=logdir,
        tensorboard_dir=tbdir,
        batch_size=FLAGS.batch_size * 4,
        visualization=False,
        new=(not exists_checkpoint),
        patience=FLAGS.patience,
        num_workers=FLAGS.num_workers,
        k=[1, None, 'max_bin_count', FLAGS.batch_size][FLAGS.k] if FLAGS.k < 4 else FLAGS.k,
        return_tester=False,
        multimodal=FLAGS.multimodal,
        splits_to_eval=[('val', ['train'])],
        primary_metric=FLAGS.primary_metric,
    )

    # create the trainer
    trainer = create_trainer(
        models=models,
        optimizers=optimizers,
        batch_size=FLAGS.batch_size,
        iterations_per_epoch=FLAGS.num_iters_per_epoch,
        loss_funcs=loss_funcs,
        mining_funcs=mining_funcs,
        sampler=sampler,
        dataset=train_dataset,
        hooks=hooks,
        end_of_epoch_hook=end_of_epoch_hook,
        lr_schedulers=lr_schedulers,
        num_workers=FLAGS.num_workers,
        loop=True,
        multimodal=FLAGS.multimodal,
    )
    logging.info('Network setup completed successfully!')

    resume_epoch = hooks.load_latest_saved_models(
        trainer, modeldir, device=device, best=False)

    # load from last checkpoint if set
    resume_epoch = None
    if exists_checkpoint:
        try:
            resume_epoch = load_checkpoint(
                experiment_dir, trunk, embedder, optimizers, lr_schedulers,
                loss_funcs, mining_funcs, multimodal=FLAGS.multimodal,
                device=device)
        except FileNotFoundError:
            logging.error('Failed to find checkpoint. Starting from scratch')

    # launch tensorboard
    if FLAGS.tb:
        tb_log_path = os.path.join(experiment_dir, 'tensorboard')
        launch_tensorboard(tb_log_path)

    # start training only embedder with frozen trunk
    if resume_epoch in [None, 1]:
        epoch = 1
        if FLAGS.warm_start:
            trainer.freeze_these = [models['trunk'].parameters()]
            trainer.train(start_epoch=epoch, num_epochs=1)
            epoch += 1
    else:
        epoch = resume_epoch

    # continue both embedder and trunk
    trainer.freeze_these = ()
    trainer.train(start_epoch=epoch, num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    FLAGS(sys.argv)  # initialize command line arguments
    main()
