import os
import sys
import logging
import torch
from math import radians, degrees
from tqdm import tqdm

from pytorch_metric_learning.testers import GlobalTwoStreamEmbeddingSpaceTester
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions
from .logging_utils import visualizer_hook, get_logging_hooks

sys.path.insert(1, '../src')
from loss_and_miner_utils.quaternion import tensor_euler_to_quaternion


class MultimodalTwoStreamEmbeddingSpaceTester(GlobalTwoStreamEmbeddingSpaceTester):
    """ Tester for double stream multi-modal metric learning """
    def compute_all_embeddings(
            self, dataloader, trunk_model, embedder_model, split=''):
        start, end = 0, 0
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader, desc=split)):
                anchors, posnegs, label = self.data_and_label_getter(data)
                label = common_functions.process_label(
                    label, self.label_hierarchy_level, self.label_mapper
                )
                anchor_embeddings = self.get_embeddings_for_eval(trunk_model, embedder_model, anchors, stream=1)
                posneg_embeddings = self.get_embeddings_for_eval(trunk_model, embedder_model, posnegs, stream=2)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if idx == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_anchors = torch.zeros(len(dataloader.dataset), anchor_embeddings.size(1))
                    all_posnegs = torch.zeros(len(dataloader.dataset), posneg_embeddings.size(1))

                end = start + posneg_embeddings.size(0)
                all_anchors[start:end] = anchor_embeddings
                all_posnegs[start:end] = posneg_embeddings
                labels[start:end] = label
                start = end
        return all_anchors, all_posnegs, labels

    def compute_db_embeddings(
            self, dataloader, trunk_model, embedder_model, split=''):
        start, end = 0, 0
        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataloader, desc=split)):
                samples, label = self.data_and_label_getter(data)
                label = common_functions.process_label(
                    label, self.label_hierarchy_level, self.label_mapper
                )
                embeddings = self.get_embeddings_for_eval(trunk_model, embedder_model, samples, stream=2)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if idx == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_embeddings = torch.zeros(len(dataloader.dataset), embeddings.size(1))

                end = start + embeddings.size(0)
                all_embeddings[start:end] = embeddings
                labels[start:end] = label
                start = end
        return all_embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs, stream):
        input_imgs = common_functions.to_device(
            input_imgs, device=self.data_device, dtype=self.dtype
        )
        trunk_output = trunk_model(input_imgs, stream)
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output, stream)

    def set_reference_and_query(
            self, embeddings_and_labels, query_split_name, reference_split_names
            ):
        assert (len(reference_split_names) == 1), "reference_split_names must contain only one name"
        reference_split_name = reference_split_names[0]
        query_embeddings, query_labels = embeddings_and_labels[query_split_name]
        ref_embeddings, ref_labels = embeddings_and_labels[reference_split_name]

        query_half = int(query_embeddings.shape[0] / 2)
        ref_half = int(ref_embeddings.shape[0] / 2)

        anchors_embeddings = query_embeddings[:query_half]
        posneg_embeddings = ref_embeddings[ref_half:]

        anchor_labels = query_labels[:query_half]
        posneg_labels = ref_labels[ref_half:]

        return anchors_embeddings, anchor_labels, posneg_embeddings, posneg_labels

    def do_knn_and_accuracies(
        self, accuracies, embeddings_and_labels, query_split_name, reference_split_names
    ):
        (
            query_embeddings,
            query_labels,
            reference_embeddings,
            reference_labels,
        ) = self.set_reference_and_query(
            embeddings_and_labels, query_split_name, reference_split_names
        )
        self.label_levels = [0]

        a = self.accuracy_calculator.get_accuracy(
            query_embeddings,
            reference_embeddings,
            query_labels,
            reference_labels,
            self.embeddings_come_from_same_source(
                query_split_name, reference_split_names
            ),
        )
        for metric, v in a.items():
            keyname = self.accuracies_keyname(metric, label_hierarchy_level=0)
            accuracies[keyname] = v


class CustomAccuracyCalculator(accuracy_calculator.AccuracyCalculator):

    def get_accuracy(self, query, reference, query_labels, reference_labels,
                     embeddings_come_from_same_source, include=(), exclude=()):
        """ Sets embeddings from same source to false to solve no metrics bug
        """
        logging.info('Calculating accuracy')
        return super().get_accuracy(
            query, reference, query_labels, reference_labels,
            embeddings_come_from_same_source=False,
            include=include, exclude=exclude)

    def calculate_nfp(self, knn_labels, query_labels, **kwargs):
        """
        Calculate percentage of negatives that are farther away from the
        anchor than the positives abbreviated as NFP
        """
        knn_distances = kwargs['knn_distances']
        percentages = torch.zeros_like(query_labels, dtype=torch.float)
        n, m = knn_distances.shape[:2]
        for i in range(n):
            indices = torch.where(knn_labels[i] == query_labels[i])[0]
            if len(indices) == 0:
                continue
            idx = indices[0]
            percentages[i] = (m-idx) / m

        avg_nfp = percentages.mean()
        return avg_nfp.item()

    def calculate_recall_at_1(self, knn_labels, query_labels, **kwargs):
        hits = torch.count_nonzero(knn_labels[:, 0] == query_labels)
        recall_at_1 = hits / len(query_labels)
        return recall_at_1.item()

    def calculate_recall_at_5(self, knn_labels, query_labels, **kwargs):
        top_5 = knn_labels[:, :5]
        query_col = query_labels.unsqueeze(1)
        in_top_5 = torch.count_nonzero(top_5 == query_col, axis=1)
        count_in_top_5 = torch.sum(in_top_5 > 0)
        recall_at_5 = count_in_top_5 / len(query_labels)
        return recall_at_5.item()

    def calc_label_distances(
            self, knn_labels, query_labels):
        closest = knn_labels[:, 0]
        if query_labels.shape[1] == 3:
            query_labels = tensor_euler_to_quaternion(query_labels, 'zyx')
            closest = tensor_euler_to_quaternion(closest, 'zyx')
        pair_dots = torch.mul(
            query_labels[:, :4], closest[:, :4]).sum(axis=1)
        pair_dots = torch.clamp(pair_dots, -1, 1).abs()
        distances = 2 * torch.arccos(pair_dots)
        return distances

    def calc_thresholded_accuracy(
            self, knn_labels, query_labels, reference_labels, threshold,
            **kwargs):
        """
        Calculates the accuracy with <threshold> degrees resolution

        Args:
            knn_labels (torch.tensor): Contains the sorted matches from knn
            query_labels (torch.tensor): Contains the labels of the queries
            reference_labels (torch.tensor): Contains the labels of the references
            threshold (int): Accuracy threshold in degrees

        Returns:
            int: accuracy with <threshold> degrees resolution
        """
        distances = self.calc_label_distances(knn_labels, query_labels)
        valid = torch.count_nonzero(distances < radians(threshold)).item()
        num_queries = len(query_labels)
        return valid / num_queries

    def calculate_accuracy_at_10(self, **kwargs):
        """
        Calculates the accuracy with 10 degrees resolution

        Returns:
            float: accuracy with 10 degrees resolution
        """
        return self.calc_thresholded_accuracy(threshold=10, **kwargs)

    def calculate_accuracy_at_30(self, **kwargs):

        """
        Calculates the accuracy with 30 degrees resolution

        Returns:
            float: accuracy with 30 degrees resolution
        """
        return self.calc_thresholded_accuracy(threshold=30, **kwargs)

    def calculate_median_error(
            self, knn_labels, query_labels, reference_labels, **kwargs):

        """
        Calculates the median angle error

        Returns:
            float: median error
        """
        distances = self.calc_label_distances(knn_labels, query_labels)
        return degrees(torch.median(distances).item())

    def requires_knn(self):
        return super().requires_knn() + ['nfp', 'recall_at_1', 'recall_at_5',
                                         'accuracy_at_10', 'accuracy_at_30',
                                         'median_error']


def create_tester(
        dataset_dict,
        model_folder='./saved_models',
        log_dir='./logs',
        tensorboard_dir='./tensorboard',
        batch_size=32,
        visualization=False,
        new=True,
        patience=5,
        k=10,
        num_workers=os.cpu_count(),
        return_tester=False,
        multimodal=False,
        splits_to_eval=None,
        primary_metric='precision_at_1',
        ):
    # importing umap is really slow so it's done only if necessary
    if visualization:
        import umap.umap_ as umap

    # create the logging hooks
    hooks = get_logging_hooks(
        csv_folder=log_dir,
        tensorboard_dir=tensorboard_dir,
        is_new_experiment=new,
        save_lists=True,
        primary_metric=primary_metric,
    )
    if primary_metric.startswith(('accuracy_at_', 'median_error')):
        metrics_to_exclude = (
            'recall_at_1', 'recall_at_5', 'nfp', 'precision_at_1',
            'r_precision', 'AMI', 'NMI', 'mean_average_precision',
            'mean_average_precision_at_r')
    elif 'precision' in primary_metric or 'recall' in primary_metric:
        metrics_to_exclude = (
            'AMI', 'NMI', 'accuracy_at_10', 'accuracy_at_30', 'median_error')
    else:
        raise ValueError(
            'Primary metric should be accuracy. median_error, or precision')

    # select multimodal or unimodal tester
    Tester = MultimodalTwoStreamEmbeddingSpaceTester if multimodal else GlobalTwoStreamEmbeddingSpaceTester

    """ Create the tester """
    tester = Tester(
        # normalize embdeddings to Euclidean norm L1 before nearest neighbors are computed.
        normalize_embeddings=True,

        # use output of trunk model to calculate nearest neighbours, instead of embedder's
        use_trunk_output=False,

        # How many dataset samples to process at each iteration when computing embeddings.
        batch_size=batch_size,

        # How many processes the dataloader will use.
        dataloader_num_workers=num_workers,

        # Number of PCA dimensions. The default is None, meaning PCA will not be applied.
        pca=None,

        # Which gpu to use for the loaded dataset samples. None->gpu/cpu, whichever is available
        data_device=None,

        # The type that the dataset output will be converted. None means no type casting
        dtype=None,

        # takes output of dataset and returns a tuple of (data,labels)
        data_and_label_getter=None,

        # in case of multiple labels per sample, select which label to use
        label_hierarchy_level='all',

        # custom dataset labels, can be 1-d or 2-d, see also label_hierarchy
        dataset_labels=dataset_dict['val'].labels,

        # maps labels to numbers 0,1,2,3...
        set_min_label_to_zero=False,

        # custom accuracy calculator extending class AccuracyCalculator
        # accuracy_calculator=None,
        accuracy_calculator=CustomAccuracyCalculator(
            k=k, exclude=metrics_to_exclude),

        # implementation with fit_transform method eg. umap
        visualizer=umap.UMAP(n_neighbors=2) if visualization else None,

        # main reason is to save plots of embeddings
        visualizer_hook=[None, visualizer_hook][visualization],

        # function logs data at the end of testing
        end_of_testing_hook=hooks.end_of_testing_hook,
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester,
        dataset_dict,
        # The folder in which to save models, optimizers etc.
        model_folder,
        # Every how many epochs to perform validation.
        test_interval=1,
        # early stopping if enough epochs have passed
        patience=patience,
        splits_to_eval=splits_to_eval,
        # splits_to_eval=[('val', ['train'])],
    )

    if return_tester:
        return hooks, end_of_epoch_hook, tester
    else:
        return hooks, end_of_epoch_hook
