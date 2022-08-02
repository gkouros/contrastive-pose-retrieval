from torch import optim


def create_optimizers(multimodal=False, **kwargs):
    if multimodal:
        # TODO fix
        return create_unimodal_optimizers(**kwargs)
        # return get_multimodal_optimizers(**kwargs)
    else:
        return create_unimodal_optimizers(**kwargs)


def create_unimodal_optimizers(
        models,
        optimizer=optim.Adam,
        trunk_lr=4e-5,
        embedder_lr=1e-4,
        trunk_wd=4e-5,
        embedder_wd=1e-3,
        scheduler='plateau',
        patience=10,
        lr_decay=0.999,
        ):
    """
    Creates optimizers for a unimodal metric learning architecture

    Args:
        models (dict): Contains the parts of the network
        optimizer (optim): The type of optimizer to use
        trunk_lr (float): The learning rate of the trunk
        embedder_lr (float): The learning rate of the embedder
        trunk_wd (float): Weight decay for the trunk part of the model
        embedder_wd (float): Weight decay for the embedder part of the model
        patience (int): How many epochs to way before decaying learning rate
        lr_decay (float): The decay of the learning rate of ADAM

    Returns:
        dict: Contains the created optimizers and lr schedulers
    """
    # create optimizers
    trunk_optimizer = optimizer(
        params=models['trunk'].parameters(),
        lr=trunk_lr, weight_decay=trunk_wd, eps=0.01)
    embedder_optimizer = optimizer(
        params=models['embedder'].parameters(),
        lr=embedder_lr, weight_decay=embedder_wd, eps=0.01)
    optimizers = {
        'trunk_optimizer': trunk_optimizer,
        'embedder_optimizer': embedder_optimizer
    }

    # create lr schedulers
    trunk_lr_scheduler = get_lr_scheduler(
        trunk_optimizer, scheduler, patience, lr_decay)
    embedder_lr_scheduler = get_lr_scheduler(
        embedder_optimizer, scheduler, patience, lr_decay)
    schedulers = {
        'trunk_scheduler_by_epoch': trunk_lr_scheduler,
        'embedder_scheduler_by_epoch': embedder_lr_scheduler
    }

    return (optimizers, schedulers)


def create_multimodal_optimizers(
        models,
        optimizer=optim.Adam,
        trunk_lr=4e-5,
        embedder_lr=1e-4,
        trunk_wd=4e-5,
        embedder_wd=1e-3,
        scheduler='plateau',
        patience=10,
        lr_decay=0.999,
        ):
    """
    Creates optimizers for a multimodal metric learning architecture

    Args:
        models (dict): Contains the parts of the network
        optimizer (optim): The type of optimizer to use
        trunk_lr (float): The learning rate of the trunk
        embedder_lr (float): The learning rate of the embedder
        trunk_wd (float): Weight decay for the trunk part of the model
        embedder_wd (float): Weight decay for the embedder part of the model
        patience (int): How many epochs to way before decaying learning rate
        lr_decay (float): The decay of the learning rate of ADAM

    Returns:
        dict: Contains the created optimizers and lr schedulers
    """
    # create optimizers
    anchor_trunk_optimizer = optimizer(
        params=models['trunk'].anchor_model.parameters(),
        lr=trunk_lr, weight_decay=trunk_wd)
    anchor_embedder_optimizer = optimizer(
        params=models['embedder'].anchor_model.parameters(),
        lr=embedder_lr, weight_decay=embedder_wd)
    posneg_trunk_optimizer = optimizer(
        params=models['trunk'].posneg_model.parameters(),
        lr=trunk_lr, weight_decay=trunk_wd)
    posneg_embedder_optimizer = optimizer(
        params=models['embedder'].posneg_model.parameters(),
        lr=embedder_lr, weight_decay=embedder_wd)

    # create lr schedulers
    anchor_trunk_lr_scheduler = get_lr_scheduler(
        anchor_trunk_optimizer, scheduler, patience, lr_decay)
    anchor_embedder_lr_scheduler = get_lr_scheduler(
        anchor_embedder_optimizer, scheduler, patience, lr_decay)
    posneg_trunk_lr_scheduler = get_lr_scheduler(
        posneg_trunk_optimizer, scheduler, patience, lr_decay)
    posneg_embedder_lr_scheduler = get_lr_scheduler(
        posneg_embedder_optimizer, scheduler, patience, lr_decay)

    return {
        'anchor_trunk_optimizer': anchor_trunk_optimizer,
        'anchor_embedder_optimizer': anchor_embedder_optimizer,
        'posneg_embedder_optimizer': posneg_embedder_optimizer,
        'posneg_embedder_optimizer': posneg_embedder_optimizer
    }, {
        'anchor_trunk_scheduler_by_plateau': anchor_trunk_lr_scheduler,
        'anchor_embedder_scheduler_by_plateau': anchor_embedder_lr_scheduler,
        'posneg_trunk_scheduler_by_plateau': posneg_trunk_lr_scheduler,
        'posneg_embedder_scheduler_by_plateau': posneg_embedder_lr_scheduler
    }


def get_lr_scheduler(
        optimizer,
        scheduler='plateau',
        patience=20,
        lr_decay=0.2,
        verbose=True,
        ):
    assert scheduler in ['none', 'exp', 'plateau', 'step']
    if scheduler == 'none':
        return None
    elif scheduler == 'exp':
        return optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=lr_decay,
            last_epoch=-1,
            verbose=True,
        )
    elif scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,  # the optimizer whose learning rate to control
            mode='max',  # scheduler activates when monitored metric plateaus
            factor=lr_decay,  # multiplier of learning rate
            patience=patience,  # how many epochs of no improvement to wait
            threshold=1e-4,  # minimum change to take into consideration
            threshold_mode='rel',  # best * ( 1 - threshold )
            cooldown=0,  # number of epochs before resuming normal operation
            min_lr=1e-7,  # minimum learning rate
            eps=1e-08,  # minimum threshold in updating lr
            verbose=True,
        )
    elif scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer,  # the optimizer whose learning rate to control
            step_size=patience,
            gamma=lr_decay,
            verbose=True,
        )
    else:
        raise ValueError(f'Scheduler {scheduler} does not exist')
