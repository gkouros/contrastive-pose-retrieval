import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedSampler(Sampler):
    """
    Copied from the official repo of ProxyAnchor loss:
    https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/dataset/sampler.py
    """
    def __init__(self, dataset, batch_size, images_per_class=3):
        self.dataset = dataset
        self.labels = dataset.labels
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.labels)
        self.num_classes = dataset._num_classes

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.labels) == sampled_classes[i])[0]
                if len(ith_class_idxs) == 0:
                    continue
                class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)
