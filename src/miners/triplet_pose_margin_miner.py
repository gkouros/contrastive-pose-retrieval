import sys
import torch
import logging
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

sys.path.insert(1, '../src')
from loss_and_miner_utils.quaternion import tensor_euler_to_quaternion


class TripletPoseMarginMiner(BaseTupleMiner):

    def __init__(self, margin=0.1, pose_positive_threshold=0.1,
                 type_of_triplets="all", pose_representation='quat', **kwargs):
        super().__init__(**kwargs)
        assert pose_representation in [
            'discrete', 'euler', 'quat', 'quat_and_model']
        self.margin = margin
        self.pose_positive_threshold = pose_positive_threshold
        self.pose_representation = pose_representation
        self.type_of_triplets = type_of_triplets
        self.pose_size = \
            4 if pose_representation in ['quat', 'quat_and_model'] else 3

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        # logging.info((labels.shape, ref_labels.shape))

        # get embedding distances
        mat = self.distance(embeddings, ref_emb)

        # form triplets based on pose distance and embedding distance
        a, p, n, pose_dists = self.get_all_triplets_indices(labels, ref_labels)

        # get positive and negative pairs
        pos_pairs = mat[a, p]
        neg_pairs = mat[a, n]

        # calculate margins between positive and negative pairs
        triplet_margin = pos_pairs - neg_pairs if self.distance.is_inverted \
            else neg_pairs - pos_pairs

        # find which triplets violate the margin
        # triplet_mask = triplet_margin <= self.margin

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin  # * pose_dists
        else:
            threshold_condition = triplet_margin <= self.margin  # * pose_dists
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        # return the triplets that violate the margin
        return (
            a[threshold_condition],
            p[threshold_condition],
            n[threshold_condition],
        )

    def calc_angular_distances(self, labels, ref_labels):
        if self.pose_representation not in [
                'discrete', 'euler', 'quat', 'quat_and_model']:
            raise ValueError('Invalid pose label representation')

        if self.pose_representation in ['discrete', 'euler']:
            # convert to quaternion representation
            qlabels = tensor_euler_to_quaternion(labels, 'zyx')
            qref_labels = tensor_euler_to_quaternion(labels, 'zyx')
        else:
            qlabels, qref_labels = labels, ref_labels

        dot_product = qlabels @ qref_labels.T
        distances = 2 * torch.acos(dot_product)

        return distances

    def get_all_triplets_indices(self, labels, ref_labels):
        # if no reference labels given use the query labels
        if ref_labels is None:
            ref_labels = labels

        # calculate pose distances between samples
        pose_dists = self.calc_angular_distances(labels, ref_labels)

        # find the matching samples
        matches = (pose_dists < self.pose_positive_threshold)

        # find the non-matching samples
        diffs = matches.byte() ^ 1

        # ignore matches of samples to themselves
        if ref_labels is labels:
            matches.fill_diagonal_(0)

        # find and return triplets
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        a, p, n = torch.where(triplets)
        return a, p, n, pose_dists[a, n]

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = self.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(
                embeddings, labels[:, :self.pose_size],
                ref_emb, ref_labels[:, :self.pose_size])
        self.output_assertion(mining_output)
        return mining_output

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            ref_labels = c_f.to_device(ref_labels, ref_emb)
        else:
            ref_emb, ref_labels = embeddings, labels
        check_shapes(ref_emb, ref_labels)
        return ref_emb, ref_labels


def check_shapes(embeddings, labels):
    if embeddings.size(0) != labels.size(0):
        raise ValueError("Number of embeddings must equal number of labels")
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
        )
    # if labels.ndim != 1:
    #     raise ValueError("labels must be a 1D tensor of shape (batch_size,)")
