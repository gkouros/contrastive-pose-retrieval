import os
from pytorch_metric_learning.trainers import TwoStreamMetricLoss
from tqdm.auto import tqdm
import logging
import torch
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class MultimodalTwoStreamMetricLoss(TwoStreamMetricLoss):
    def calculate_loss(self, curr_batch):
        (anchors, posnegs), labels = curr_batch
        embeddings = (
            self.compute_embeddings(anchors, stream=1),
            self.compute_embeddings(posnegs, stream=2),
        )

        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )

    def compute_embeddings(self, data, **kwargs):
        trunk_output = self.get_trunk_output(data, **kwargs)
        embeddings = self.get_final_embeddings(trunk_output, **kwargs)
        return embeddings

    def get_final_embeddings(self, base_output, **kwargs):
        return self.models["embedder"](base_output, **kwargs)

    def get_trunk_output(self, data, **kwargs):
        data = common_functions.to_device(data, device=self.data_device, dtype=self.dtype)
        return self.models["trunk"](data, **kwargs)


    def allowed_model_keys(self):
        return [
            "trunk", "embedder",
            # "anchor_trunk", "anchor_embedder",
            # "posneg_trunk", "posneg_embedder"
        ]

    def maybe_mine_embeddings(self, embeddings, labels):
        # for both get_all_triplets_indices and mining_funcs
        # we need to clone labels and pass them as ref_labels
        # to ensure triplets are generated between anchors and posnegs
        if "tuple_miner" in self.mining_funcs:
            (anchors_embeddings, posnegs_embeddings) = embeddings
            return self.mining_funcs["tuple_miner"](
                anchors_embeddings, labels, posnegs_embeddings, labels.clone()
            )
        else:
            labels = labels.to(embeddings[0].device)
            return lmu.get_all_triplets_indices(labels, labels.clone())


class UnimodalTwoStreamMetricLoss(TwoStreamMetricLoss):
    def maybe_mine_embeddings(self, embeddings, labels):
        # for both get_all_triplets_indices and mining_funcs
        # we need to clone labels and pass them as ref_labels
        # to ensure triplets are generated between anchors and posnegs
        if "tuple_miner" in self.mining_funcs:
            (anchors_embeddings, posnegs_embeddings) = embeddings
            return self.mining_funcs["tuple_miner"](
                anchors_embeddings, labels, posnegs_embeddings, labels.clone()
            )
        else:
            labels = labels.to(embeddings[0].device)
            return lmu.get_all_triplets_indices(labels, labels.clone())



def create_trainer(
        models,
        optimizers,
        batch_size,
        loss_funcs,
        mining_funcs,
        sampler,
        dataset,
        hooks,
        end_of_epoch_hook,
        lr_schedulers=None,
        num_workers=os.cpu_count(),
        loop=True,
        multimodal=False,
        iterations_per_epoch=0,
        ):

    # choose between multimodal and unimodal trainer
    if multimodal:
        Trainer = MultimodalTwoStreamMetricLoss
    else:
        Trainer = UnimodalTwoStreamMetricLoss

    trainer = Trainer(
        models=models,
        optimizers=optimizers,
        batch_size=batch_size,
        loss_funcs=loss_funcs,
        iterations_per_epoch=iterations_per_epoch,
        mining_funcs=mining_funcs,
        dataset=dataset,
        sampler=sampler,
        freeze_trunk_batchnorm=True,
        set_min_label_to_zero=False,
        dataloader_num_workers=num_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
        lr_schedulers=lr_schedulers,
        label_hierarchy_level='all',
    )

    return trainer
