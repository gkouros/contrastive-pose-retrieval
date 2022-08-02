import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from pytorch_metric_learning.utils import common_functions as c_f


# modified from
# https://github.com/KevinMusgrave/pytorch-metric-learning/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
class NonNeighbouringClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class of batch_size//m
    classes that have a label distance greater than 1.
    """

    def __init__(self, labels, m, batch_size, dist_threshold=1, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.dist_threshold = dist_threshold
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels) # TODO  May have to clip to
        self.list_size = length_before_new_iter

        assert (batch_size % m) == 0, \
            "batch size must be divisible by m"
        assert self.batch_size <= self.list_size, \
            "batch size must be less or equal to list size"
        assert (self.length_of_single_pass >= self.batch_size), \
            "m * (number of unique labels) must be >= batch_size"
        assert (self.batch_size % self.m_per_class) == 0, \
            "m_per_class must divide batch_size without any remainder"
        self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.list_size // self.batch_size

        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.labels)
            curr_label_set = self.get_non_neighbouring_labels()

            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = \
                    c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class

        return iter(idx_list)

    def get_non_neighbouring_labels(self):
        """
        Generates bs/m random labels from the label set
        """
        curr_labels = []
        num_curr_labels = self.batch_size // self.m_per_class
        count = 0

        while len(curr_labels) < num_curr_labels:
            random_label = np.random.choice(list(set(self.labels)))
            neighbouring = False

            for label in curr_labels:
                if abs(label - random_label) <= self.dist_threshold:
                    neighbouring = True
                    break

            if not neighbouring:
                curr_labels.append(random_label)

            count += 1
            if count > 1e6:
                raise Exception("Too many iterations while selecting labels")

        return curr_labels