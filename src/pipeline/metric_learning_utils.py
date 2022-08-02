import sys
import math
import torch

# local imports
sys.path.insert(1, '../src')
import losses as local_losses
import samplers as local_samplers
import miners as local_miners

from pytorch_metric_learning import losses, miners, samplers, distances

NUM_SAMPLES_PER_CLASS = 2

""" Selector """
def create_loss_miner_sampler(loss_type, **kwargs):
    if loss_type == 'contrastive':  # Contrastive Loss
        return get_contrastive_loss_funcs(**kwargs)
    if loss_type == 'triplet':  # Triplet Margin Loss
        return get_triplet_loss_funcs(**kwargs)
    elif loss_type == 'margin':  # Margin + Distance Weighted Sampling
        return get_margin_loss_funcs(**kwargs)
    elif loss_type == 'proxy':  # Proxy Anchor Loss - 3x faster than triplet
        return get_proxy_anchor_loss_funcs(**kwargs)
    elif loss_type == 'soft':  # Soft Triple Loss (2020)
        return get_soft_triple_loss_funcs(**kwargs)
    elif loss_type == 'arcface':  # ArcFace loss is state-of-the-art
        return get_arcface_loss_funcs(**kwargs)
    elif loss_type == 'weighted_triplet':  # Weighted Triplet Loss
        return get_weighted_triplet_loss_funcs(**kwargs)
    elif loss_type == 'weighted_contrastive':  # Weighted Contrastive Loss
        return get_weighted_contrastive_loss_funcs(**kwargs)
    elif loss_type == 'dynamic_margin':  # Dynamic Margin Loss
        return get_dynamic_margin_loss_funcs(**kwargs)
    else:
        raise ValueError(f'Loss type {loss_type} does not exist')


""" Triplet Loss with (Semi-)Hard Negative Mining """
def get_contrastive_loss_funcs(
        dataset,
        batch_size=32,
        pos_margin=0,
        neg_margin=1,
        type_of_triplets='semihard',
        use_miner=True,
        **kwargs,
        ):

    # Set the loss function
    loss = losses.ContrastiveLoss(
        pos_margin=pos_margin,
        neg_margin=neg_margin,
        distance=distances.LpDistance(),
    )

    # Set the mining function
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(
        labels=dataset.labels,
        m=2,
        length_before_new_iter=len(dataset),
    )

    return (
        {'metric_loss': loss},
        {'tuple_miner': miner} if use_miner else {},
        sampler
    )

""" Triplet Loss with (Semi-)Hard Negative Mining """
def get_triplet_loss_funcs(
        dataset,
        batch_size=32,
        margin=0.2,
        type_of_triplets='semihard',
        use_nnc_sampler=False,
        **kwargs,
        ):

    # Set the loss function
    loss = losses.TripletMarginLoss(
        margin=margin,
        swap=True,
        smooth_loss=True,  # new_loss = log(1 + exp(loss))
        distance=distances.CosineSimilarity(),
    )

    # Set the mining function
    miner = miners.TripletMarginMiner(
        margin=margin,
        type_of_triplets=type_of_triplets  # all, hard, semihard or easy
    )
    # miner = miners.MultiSimilarityMiner(epsilon=margin)

    # Set the dataloader sampler
    if use_nnc_sampler:
        sampler = local_samplers.NonNeighbouringClassSampler(
            labels=dataset.labels,
            m=NUM_SAMPLES_PER_CLASS,  # num of samples per class
            dist_threshold=1,
            batch_size=batch_size,
            length_before_new_iter=len(dataset),
        )
    else:
        sampler = samplers.MPerClassSampler(
            labels=dataset.labels,
            m=2,
            length_before_new_iter=len(dataset),
        )

    return {'metric_loss': loss}, {'tuple_miner': miner}, sampler


""" Weighted Margin Loss for pose estimation datasets """
def get_weighted_triplet_loss_funcs(
        dataset,
        batch_size=32,
        margin=0.2,
        type_of_triplets='semihard',
        use_miner=True,
        labelling_method='azim',
        pose_positive_threshold=0.1,
        use_dynamic_margin=True,
        **kwargs,
        ):

    # Set the loss function
    loss = local_losses.WeightedMarginTripletLoss(
        margin=margin,
        swap=True,
        smooth_loss=True,  # new_loss = log(1 + exp(loss))
        distance=distances.CosineSimilarity(),
        mode=labelling_method,
        use_dynamic_margin=use_dynamic_margin,
    )
    if labelling_method in ['azim', 'discrete']:
        # Set the mining function
        miner = miners.TripletMarginMiner(
            margin=margin,
            type_of_triplets=type_of_triplets  # all, hard, semihard or easy
        )

        # miner = miners.MultiSimilarityMiner(epsilon=margin)
        sampler = samplers.MPerClassSampler(
            labels=dataset.labels,
            m=2,
            length_before_new_iter=len(dataset),
        )
    elif labelling_method in ['euler', 'quat']:
        miner = local_miners.TripletPoseMarginMiner(
            margin=margin,
            pose_representation=labelling_method,
            pose_positive_threshold=pose_positive_threshold,
        )
        sampler = local_samplers.PoseBasedSampler(
            dataset,
            batch_size=batch_size,
            images_per_class=2,
            pose_positive_threshold=pose_positive_threshold,
        )
    else:
        raise ValueError(
            'Invalid labelling method provided: %s' % labelling_method)

    return (
        {'metric_loss': loss},
        {'tuple_miner': miner} if use_miner else {},
        sampler
    )

""" Weighted Margin Loss for pose estimation datasets """
def get_weighted_contrastive_loss_funcs(
        dataset,
        batch_size=32,
        margin=0.2,
        type_of_triplets='semihard',
        use_miner=True,
        labelling_method='azim',
        pose_positive_threshold=0.1,
        use_dynamic_margin=True,
        **kwargs,
        ):

    # Set the loss function
    loss = local_losses.WeightedMarginContrastiveLoss(
        margin=margin,
        distance=distances.LpDistance(),
        mode=labelling_method,
        use_dynamic_margin=use_dynamic_margin,
    )
    if labelling_method in ['azim', 'nemo']:
        # Set the mining function
        miner = miners.TripletMarginMiner(
            margin=margin,
            type_of_triplets=type_of_triplets  # all, hard, semihard or easy
        )
        # miner = miners.MultiSimilarityMiner(epsilon=margin)
        sampler = samplers.MPerClassSampler(
            labels=dataset.labels,
            m=2,
            length_before_new_iter=len(dataset),
        )
    elif labelling_method in ['euler', 'quat', 'discrete']:
        miner = local_miners.TripletPoseMarginMiner(
            margin=margin,
            pose_representation=labelling_method,
            pose_positive_threshold=pose_positive_threshold,
        )
        sampler = local_samplers.PoseBasedSampler(
            dataset,
            batch_size=batch_size,
            images_per_class=2,
            pose_positive_threshold=pose_positive_threshold,
        )
    else:
        raise ValueError(
        'Invalid labelling method provided: %s' % labelling_method)


    return (
        {'metric_loss': loss},
        {'tuple_miner': miner} if use_miner else {},
        sampler
    )

""" Weighted Margin Loss for pose estimation datasets """
def get_dynamic_margin_loss_funcs(
        dataset,
        batch_size=32,
        margin=0.2,
        type_of_triplets='semihard',
        use_miner=True,
        pose_positive_threshold=0.1,
        **kwargs,
        ):

    # Set the loss function
    loss = local_losses.DynamicMarginLoss(
        margin=margin,
        swap=True,
        smooth_loss=True,  # new_loss = log(1 + exp(loss))
        distance=distances.CosineSimilarity()
    )

    miner = local_miners.TripletPoseMarginMiner(
        margin=margin,
        pose_representation='quat_and_model',
        pose_positive_threshold=pose_positive_threshold,
    )
    sampler = local_samplers.PoseBasedSampler(
        dataset,
        batch_size=batch_size,
        images_per_class=2,
        pose_positive_threshold=pose_positive_threshold,
    )

    return (
        {'metric_loss': loss},
        {'tuple_miner': miner} if use_miner else {},
        sampler
    )



""" Margin Loss with Distance Weighting Sampling """
def get_margin_loss_funcs(
        dataset,
        batch_size=32,
        embedding_size=128,
        margin=0.2,
        type_of_triplets='semihard',
        **kwargs,
        ):

    # Set the loss function
    loss = losses.MarginLoss(
        margin=margin,
        nu=0,
        beta=1.2,
        triplets_per_anchor=type_of_triplets,
        learn_beta=False,
        num_classes=None,
    )

    # Set the mining function
    miner = miners.UniformHistogramMiner(
        num_bins=100,
        pos_per_bin=10,
        neg_per_bin=10,
    )

    sampler = samplers.MPerClassSampler(
        labels=dataset.labels,
        m=2,
        length_before_new_iter=len(dataset)
    )

    return {'metric_loss': loss}, {'tuple_miner': miner}, sampler


""" ProxyAnchor Loss with semi-hard sampling """
def get_proxy_anchor_loss_funcs(
        dataset,
        batch_size=32,
        embedding_size=512,
        margin=0.1,
        alpha=32,
        **kwargs,
        ):

    # create loss function
    loss = losses.ProxyAnchorLoss(
        embedding_size=embedding_size,
        margin=margin,
        num_classes=dataset._num_classes,
        alpha=alpha,
    )

    # create optimizer for the loss function
    loss_optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)

    # create sampler
    sampler = local_samplers.BalancedSampler(
        dataset, batch_size, NUM_SAMPLES_PER_CLASS)

    return (
        {'metric_loss': loss, 'metric_loss_optimizer': loss_optimizer},
        {},
        sampler,
    )


""" SoftTriple Loss with semi-hard sampling """
def get_soft_triple_loss_funcs(
        dataset,
        batch_size=32,
        embedding_size=128,
        margin=0.01,
        use_nnc_sampler=False,
        **kwargs,
        ):

    # create loss function
    loss = losses.SoftTripleLoss(
        embedding_size=embedding_size,
        centers_per_class=98,
        la=20,
        gamma=0.1,
        margin=0.01,
        num_classes=dataset._num_classes,
    )

    # create optimizer for the loss function
    loss_optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)

    # create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(dataset.sample_weights),
        num_samples=len(dataset)
    )

    return (
        {'metric_loss': loss, 'metric_loss_optimizer': loss_optimizer},
        # {'tuple_miner': miner},
        {},
        sampler,
    )


""" SoftTriple Loss with semi-hard sampling """
def get_arcface_loss_funcs(
        dataset,
        batch_size=32,
        embedding_size=128,
        margin=28.6,
        use_nnc_sampler=False,
        **kwargs,
        ):

    # create loss function
    loss = losses.ArcFaceLoss(
        embedding_size=embedding_size,
        num_classes=dataset._num_classes,
        margin=margin,
        # margin=360.0 / dataset._num_classes,
        scale=64,
    )

    # create optimizer for the loss function
    loss_optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)

    # create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        dataset.sample_weights,
        num_samples=len(dataset)
    )

    return (
        {'metric_loss': loss, 'metric_loss_optimizer': loss_optimizer},
        {},
        sampler,
    )
