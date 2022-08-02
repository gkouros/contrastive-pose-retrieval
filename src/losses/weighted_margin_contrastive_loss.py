import sys
import torch

from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

sys.path.insert(1, '../src')
from loss_and_miner_utils.quaternion import tensor_euler_to_quaternion



class WeightedMarginContrastiveLoss(GenericPairLoss):
    def __init__(self, mode, margin=1, use_dynamic_margin=True, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.margin = margin
        self.mode = mode
        self.use_dynamic_margin = use_dynamic_margin

        self.add_to_recordable_attributes(
            list_of_names=["margin"], is_stat=False)

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
        # indices_tuple before: N x {a, p, n} indices
        # indices_tuple after: N x {a, p, a, n} indices
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        # if no indices return 0 loss
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        # calculate matrix of embedding distances
        mat = self.distance(embeddings)
        # calculate loss using emb. distances, {a,p,a,n} pairs and labels
        return self.loss_method(mat, indices_tuple, labels)

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple, labels):
        # initialize positive and negative loss to 0
        pos_loss, neg_loss = 0, 0
        # get positive pairs from indices
        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        # get negative pairs from indices
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
        # if there are positive pairs, get positive loss
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pairs, pos_pair_dist, "pos", labels)
        # if there are negative pairs, get negative loss
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pairs, neg_pair_dist, "neg", labels)
        # return dictionary with pairs, losses, and reduction types
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pairs, pair_dists, pos_or_neg, labels):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        # margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        label_dist = self.calc_label_distance(pairs, pair_dists, labels)
        margin = self.margin * (label_dist if self.use_dynamic_margin else 1.0)
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def calc_label_distance(self, pairs, dists, labels):
        labels1 = labels[pairs[0]]
        labels2 = labels[pairs[1]]
        if self.mode == 'azim':  # assuming 36 labels
            labels *= 10
            label_dist1 = labels1 - labels2
            label_dist2 = -label_dist1
            label_dist1[label_dist1 < 0] = 360 - label_dist1[label_dist1 < 0]
            label_dist2[label_dist2 < 0] = 360 - label_dist2[label_dist2 < 0]
            label_dist = torch.minimum(label_dist1, label_dist2)
        elif self.mode in ['euler', 'quat', 'discrete']:
            if self.mode in ['euler', 'discrete']:
                labels1 = tensor_euler_to_quaternion(labels1, 'zyx')
                labels2 = tensor_euler_to_quaternion(labels2, 'zyx')
            pair_dots = torch.mul(labels1, labels2).sum(axis=1)
            pair_dots = torch.clamp(pair_dots, -1, 1).abs()
            label_dist = 2 * torch.arccos(pair_dots)
        return label_dist

    def pos_calc(self, pos_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(pos_pair_dist, margin))

    def neg_calc(self, neg_pair_dist, margin):
        return torch.nn.functional.relu(self.distance.margin(margin, neg_pair_dist))

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]

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

    def pair_based_loss(self, mat, indices_tuple, labels):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple, labels)
