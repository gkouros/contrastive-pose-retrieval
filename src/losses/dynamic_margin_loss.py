import sys
import logging
import torch
from math import pi

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.utils import common_functions as c_f

sys.path.insert(1, '../src')
from loss_and_miner_utils.quaternion import euler_to_quaternion


class DynamicMarginLoss(BaseMetricLossFunction):
    """
    Based on '3D Object Instance Recognition and Pose Estimation Using Triplet
    Loss with Dynamic Margin' -> https://arxiv.org/abs/1904.04854

    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="semihard",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=True)
        self.zero = torch.tensor(0)

    def forward(self, embeddings, labels, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        self.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        # get azim, elev and theta labels
        a_labels = labels[anchor_idx][:, :4]
        p_labels = labels[positive_idx][:, :4]
        n_labels = labels[negative_idx][:, :4]
        # get CAD model names
        a_models = labels[anchor_idx][:, 4]
        p_models = labels[positive_idx][:, 4]
        n_models = labels[negative_idx][:, 4]
        # get embedding distances
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        # use pn distance if smaller than an
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)
        pair_dots = torch.mul(a_labels, n_labels).sum(axis=1)
        pair_dots = torch.clamp(pair_dots, -1, 1).abs()
        label_dists = 2 * torch.arccos(pair_dots)

        # calculate margin based
        margin = self.margin * (label_dists * (a_models == p_models)
                                + 10 * (a_models != p_models))
        triplet_loss = torch.nn.functional.relu(
            1 - an_dists / (ap_dists + margin))
        pair_loss = ap_dists
        loss = triplet_loss + pair_loss

        if self.smooth_loss:
            loss = torch.nn.functional.softplus(loss)
        else:
            loss = torch.nn.functional.relu(loss)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def check_shapes(self, embeddings, labels):
        if embeddings.size(0) != labels.size(0):
            raise ValueError("Number of embeddings must equal number of labels")
        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a 2D tensor of shape (batch_size, embedding_size)"
            )
        # The following two lines are commented out to allow for 2D labels
        # if labels.ndim != 1:
        #     raise ValueError("labels must be a 1D tensor of shape (batch_size,)")

    def get_default_reducer(self):
        return AvgNonZeroReducer()
