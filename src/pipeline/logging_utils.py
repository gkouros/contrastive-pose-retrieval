# logging and visualization imports
import os
import time
import shutil
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import logging

import torch
import pytorch_metric_learning
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning.utils import common_functions as c_f

import tensorboard
from tensorboard.util import tb_logging
import pprint


def setup_logging(
        save_logs=True,
        logdir='./logs',
        level='info',
        ):
    """ Setup logging to file and stdout """
    if not os.path.isdir(logdir) and save_logs:
        try:
            os.makedirs(logdir)
        except FileExistsError as err:
            print(f'Failed to create log dir {logdir} with error {err}')

    handlers = [logging.StreamHandler()]
    if save_logs:
        handlers += [logging.FileHandler(os.path.join(logdir, 'main.log'))]
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers)
    logging.info("Pytorch Version%s" % torch.__version__)
    logging.info("PML Version%s" % pytorch_metric_learning.__version__)


def get_logging_hooks(
        csv_folder="./logs",
        tensorboard_dir="tensorboard",
        is_new_experiment=True,
        save_lists=True,
        primary_metric='precision_at_1',
        ):
    """
    [summary]

    Args:
        csv_folder (str, optional): Defaults to "logs".
        tensorboard_dir (str, optional): Defaults to "tensorboard".
        is_new_experiment (bool, optional): Defaults to True.
        save_lists (bool, optional): Defaults to False.

    Returns:
        [type]: [description]
    """
    # record_keeper, record_writer, tensorboard_writer = \
    record_keeper, _, _ = logging_presets.get_record_keeper(
        csv_folder=csv_folder,
        tensorboard_folder=tensorboard_dir,
        global_db_path=None,
        experiment_name=None,
        is_new_experiment=is_new_experiment,
        save_lists=save_lists,
    )

    # hooks = logging_presets.get_hook_container(
    # hooks = logging_presets.HookContainer(
    hooks = CustomHookContainer(
        record_keeper=record_keeper,
        primary_metric=primary_metric,
        validation_split_name='val',
        save_models=True,
    )

    return hooks


class CustomHookContainer(logging_presets.HookContainer):
    def save_models_and_eval(
            self,
            trainer,
            dataset_dict,
            model_folder,
            test_interval,
            tester,
            splits_to_eval=[('val', ['train'])],
            collate_fn=None,
            model_saving_frequency=1,
            ):
        epoch = int(trainer.epoch)
        accuracies = tester.test(
            dataset_dict,
            epoch,
            trainer.models["trunk"],
            trainer.models["embedder"],
            splits_to_eval,
            collate_fn,
        )[splits_to_eval[0][0]]
        logging.info('-' * 40)
        for key in accuracies:
            if key == 'epoch':
                continue
            metric = key.split('_level0')[0]
            value = accuracies[key]
            logging.info(f'{metric}: {value}')
        logging.info('-' * 40)
        prev_best_epoch, _ = self.get_best_epoch_and_accuracy(
            tester, self.validation_split_name
        )
        is_new_best, curr_accuracy, best_epoch, best_accuracy = \
            self.is_new_best_accuracy(tester, self.validation_split_name, epoch)

        self.record_keeper.save_records()
        trainer.step_lr_plateau_schedulers(curr_accuracy)
        # save latest model
        if epoch % model_saving_frequency == 0:
            logging.info('Saving model of current epoch')
            self.save_models(
                trainer, model_folder, epoch, epoch - test_interval
            )
        if is_new_best:
            logging.info(
                '\033[92m' + f'New best accuracy! {best_accuracy}\033[0m')
            curr_suffix = "best%d" % best_epoch
            prev_suffix = "best%d" % prev_best_epoch \
                if prev_best_epoch is not None else None
            logging.info('Saving best model')
            self.save_models(
                trainer, model_folder, curr_suffix, prev_suffix
            )  # save best model

        return best_epoch

    def save_models(self, trainer, model_folder, curr_suffix, prev_suffix=None):
        if self.do_save_models:
            for obj_dict in tqdm([
                getattr(trainer, x, {}) for x in self.saveable_trainer_objects
            ], desc='Saving model %s' % curr_suffix):
                c_f.save_dict_of_models(obj_dict, curr_suffix, model_folder)
                if prev_suffix is not None:
                    c_f.delete_dict_of_models(obj_dict, prev_suffix, model_folder)
                time.sleep(1)


def visualizer_hook(
        umapper,
        umap_embeddings,
        labels,
        split_name,
        keyname,
        epoch,
        **args
        ):
    """
    [summary]

    Args:
        umapper ([type]): [description]
        umap_embeddings ([type]): [description]
        labels ([type]): [description]
        split_name ([type]): [description]
        keyname ([type]): [description]
        epoch ([type]): [description]
    """
    logging.info("UMAP plot for the {} split and label set {}".format(
        split_name, keyname))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(10, 5))

    # split anchor and positive/negative embeddings
    half = int(umap_embeddings.shape[0] / 2)
    anchors = umap_embeddings[:half]
    posneg = umap_embeddings[half:]
    labels = labels[:half]

    # plot embeddings in 2D embedding space
    for i in range(num_classes):
        idx = labels == label_set[i]
        c = [random.random() for _ in range(3)]

        # plot anchors as transparent circles
        plt.plot(anchors[idx, 0], anchors[idx, 1],
                 "x", markersize=5, color=c, alpha=1)

        # plot posnegs as Xs
        plt.plot(posneg[idx, 0], posneg[idx, 1],
                 "o", markersize=5, color=c, alpha=0.3)

        # plot conection between an anchor and its corresponding positive
        plt.plot([anchors[idx, 0], posneg[idx, 0]],
                 [anchors[idx, 1], posneg[idx, 1]],
                 '--', color=c, alpha=0.3)

        plt.title(f'Epoch #{epoch}')

    plt.show()


def delete_dir(dirname):
    if os.path.exists(dirname) and os.path.isdir(dirname):
        logging.warning(f'Deleting old {dirname} directory.')
        shutil.rmtree(dirname)


def new_dir(dirname):
    delete_dir(dirname)
    os.makedirs(dirname)


def launch_tensorboard(dirname='./tensorboard', loglevel=logging.ERROR):
    logger = tb_logging.get_logger()
    logger.setLevel(loglevel)
    os.makdirs(dirname)
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', dirname])
    url = tb.launch()
    logging.info(f'Tensorboard url: {url}')


class PrettyLog():

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return pprint.pformat(self.obj)
