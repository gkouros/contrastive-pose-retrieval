#!/usr/bin/env python3
# coding: utf-8

import sys
import pickle
import logging
import torch
from absl import flags
from scipy.spatial.transform import Rotation as R
from torchvision.models import resnet50

sys.path.append(sys.path[0] + '/../src')
from pipeline.network import create_network
from pipeline.transforms import get_inference_transforms
from loss_and_miner_utils.quaternion import qeuler


torch.backends.cudnn.benchmark = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string('weights_path', '', 'The path to the model weights')
flags.DEFINE_string('image_path', '', 'The path to the query image')
flags.DEFINE_string('refset_path', '', 'The path to the reference set embeddings')

flags.mark_flags_as_required(['weights_path', 'image_path', 'refset_path'])


if __name__ == '__main__':
    # initialize command line arguments
    FLAGS(sys.argv)
    # Set the cuda device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    logging.info('device: %s' % device)
    num_workers = 8

    # create model and load weights
    models = create_network(
        device=device,
        backbone=resnet50,
        embedding_size=512,
        multimodal=True,
        pretrained=True,
    )
    trunk = models['trunk']
    embedder = models['embedder']
    load_weights(FLAGS.weights_path, trunk, embedder, multimodal=True, device=device)

    # set model to inference mode
    trunk.eval()
    embedder.eval()

    # get transforms
    transforms = get_inference_transforms(output_size=(128, 336))

    # read query image
    query_img = cv2.imread(FLAGS.image_path, -1)

    # preprocess query image
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_img = transforms(query_img)

    # encode query image
    query_emb = embedder(trunk(query[None], stream=1), stream=1)
    query_emb = torch.nn.functional.normalize(query_emb)

    # load reference embeddings
    with open(FLAGS.refset_path, 'rb') as f:
        refset = pickle.load(f)
    refset_embeddings = refset[1]  # depth / uv maps / render etc. - posnegs
    refset_labels = refset[2]

    # calculate distance of query image to reference set
    dist_mat = distance(query_emb, ref_embeddings)
    distances, indices = torch.topk(
        dist_mat, 1, largest=distance.is_inverted, dim=1)
    idx = indices[0].item()
    pose = labels[idx]

    if len(pose) == 4:
        pose = qeuler(pose, 'zyx', return_degrees=True)
    print('Predicted pose: \n\tazimuth: %f \n\televation: %f \n\ttheta: %f' % tuple(pose))

