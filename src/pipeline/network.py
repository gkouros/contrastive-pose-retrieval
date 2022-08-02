import os
import torch
from torch import nn
from torchvision import models
from pytorch_metric_learning.utils import common_functions as c_f

from pipeline.models import MLP, MultimodalModel


def create_network(
        device=torch.device('cpu'),
        backbone=models.resnet18,
        embedding_size=128,
        multimodal=False,
        pretrained=True,
        ):
    """
    [summary]

    Args:
        device ([type], optional): [description]. Defaults to device('cpu').
        backbone ([type], optional): [description]. Defaults to resnet18.
        embedding_size (int, optional): [description]. Defaults to 128.

    Returns:
        tuple: Contains the created trunk and embedder
    """
    # Set trunk model and replace the softmax layer with an identity function

    if multimodal:
        # create multimodal trunk
        anchor_trunk, trunk_output_size = create_trunk(backbone, device)
        posneg_trunk, _ = create_trunk(backbone, device)
        trunk = nn.DataParallel(MultimodalModel(anchor_trunk, posneg_trunk).to(device))

        # create multimodal embedder
        anchor_embedder = create_embedder(trunk_output_size, embedding_size, device)
        posneg_embedder = create_embedder(trunk_output_size, embedding_size, device)
        embedder = nn.DataParallel(MultimodalModel(anchor_embedder, posneg_embedder).to(device))
    else:
        # create unimodal trunk
        trunk, trunk_output_size = create_trunk(backbone, device)

        # create unimodal embedder
        embedder = create_embedder(trunk_output_size, embedding_size, device)

    return {"trunk": trunk, "embedder": embedder}


def create_trunk(backbone, device, pretrained=True, mocov2=False):
    trunk = backbone(pretrained=pretrained)
    if mocov2:
        mocov2_checkpoint = torch.load(
            '/esat/topaz/gkouros/models/res50_moco_v2_800ep_pretrain.pth',
            map_location=lambda storage, loc: storage.cuda())
        trunk.load_state_dict(mocov2_checkpoint, strict=False)
    output_size = trunk.fc.in_features
    trunk.fc = c_f.Identity()
    return nn.DataParallel(trunk.to(device)), output_size


def create_embedder(trunk_output_size, embedding_size, device):
    return nn.DataParallel(MLP([trunk_output_size, embedding_size]).to(device))


def load_checkpoint(path, trunk, embedder, optimizers, lr_schedulers,
                    loss_funcs, mining_funcs, multimodal=False,
                    device=torch.device('cpu')):
    resume_epoch, suffix = c_f.latest_version(
        os.path.join(path, 'saved_models'))
    trunk_path = f'{path}/saved_models/trunk_{resume_epoch}.pth'
    emb_path = f'{path}/saved_models/embedder_{resume_epoch}.pth'

    trunk_checkpoint = torch.load(trunk_path, map_location=device)
    embedder_checkpoint = torch.load(emb_path, map_location=device)
    trunk.module.load_state_dict(trunk_checkpoint, strict=True)
    embedder.module.load_state_dict(embedder_checkpoint, strict=True)
    if multimodal:
        trunk.module.anchor_model.module.linear = c_f.Identity()
        trunk.module.posneg_model.module.linear = c_f.Identity()
        embedder.module.anchor_model.module.linear = c_f.Identity()
        embedder.module.posneg_model.module.linear = c_f.Identity()
    else:
        trunk.module.linear = c_f.Identity()
        embedder.module.linear = c_f.Identity()

    # move models to device
    trunk.to(device)
    embedder.to(device)

    # set models to training mode
    trunk.train()
    embedder.train()

    # load optimizer and lr scheduler checkpoints
    for name in optimizers:
        chk = torch.load(f'{path}/saved_models/{name}_{resume_epoch}.pth', map_location=device)
        optimizers[name].load_state_dict(chk)
    for name in lr_schedulers:
        chk = torch.load(f'{path}/saved_models/{name}_{resume_epoch}.pth', map_location=device)
        lr_schedulers[name].load_state_dict(chk)

    # load loss and miner from checkpoint
    chk = torch.load(f'{path}/saved_models/metric_loss_{resume_epoch}.pth', map_location=device)
    loss_funcs['metric_loss'].load_state_dict(chk)
    chk = torch.load(f'{path}/saved_models/tuple_miner_{resume_epoch}.pth', map_location=device)
    mining_funcs['tuple_miner'].load_state_dict(chk)

    return resume_epoch + 1