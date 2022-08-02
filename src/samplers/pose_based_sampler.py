import math
import numpy as np
from torch.utils.data.sampler import Sampler
from scipy.spatial.transform import Rotation as R


class PoseBasedSampler(Sampler):
    """
    Copied from the official repo of ProxyAnchor loss:
    https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/dataset/sampler.py
    """
    def __init__(self, dataset, batch_size, images_per_class=2,
                 pose_positive_threshold=0.2):
        self.labels = dataset.labels
        self.size = len(dataset)
        self.labelling_method = dataset._labelling_method
        self.weights = dataset.get_weights()
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.labels)
        self.pose_positive_threshold = pose_positive_threshold
        self.matches = (self.get_label_distances(self.labels) >
                        self.pose_positive_threshold) ^ 1
        np.fill_diagonal(self.matches, 0)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = self.size // self.batch_size
        ret = []
        while num_batches > 0:
            indices = np.random.choice(len(self.labels), self.num_groups,
                                       replace=False, p=self.weights)
            for sample1 in indices:
                curr_matches = np.where(self.matches[sample1, :])[0]
                if len(curr_matches) > 0:
                    sample2 = np.random.choice(curr_matches)
                else:
                    sample2 = sample1
                ret.extend([sample1, sample2])
            num_batches -= 1

        return iter(ret)

    def get_label_distances(self, labels):
        if self.labelling_method in ['discrete', 'euler']:
            assert len(labels[0]) == 3
            qlabels = np.array([self.euler2quaternion(x) for x in labels])
        elif self.labelling_method == 'quat':
            assert len(labels[0]) == 4
            qlabels = np.array(labels)
        elif self.labelling_method == 'quat_and_model':
            assert len(labels[0]) == 5
            qlabels = np.array(labels[:, :4])
        else:
            raise ValueError('PoseBasedSampler should be called with euler or '
                             'quaternion pose representation')
        dot_product = np.dot(qlabels, qlabels.T)
        dot_product = np.clip(dot_product, -1, 1)
        distances = 2 * np.arccos(np.abs(dot_product))
        return distances

    def euler2quaternion(self, label):
        assert len(label) == 3
        azim, elev, theta = label
        rot = R.from_euler('zyx', [azim, elev, theta], degrees=False)
        label = rot.as_quat()
        return label
