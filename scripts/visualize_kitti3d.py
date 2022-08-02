# !/usr/bin/env python3
""" Visualize instances from the datsets

Use right arrow to go forward, left arrow to go back and ESC to close window.
"""
import os
import sys
import cv2
import numpy as np
import traceback
from absl import flags
import torch

# local imports
sys.path.append(sys.path[0] + '/../src')
from dataloader.kitti_object_dataset import KITTIObjectDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', '/data/datasets/KITTI3D',
                    'The location of the dataset')
flags.DEFINE_string('rendering_dir', '/data/datasets/pascal3d/',
                    'The location of the dataset')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'test'], 'Dataset split')
flags.DEFINE_enum('category', 'car', ['car'], 'Object category')
flags.DEFINE_integer('start', 0, 'Starting idx to iterate from')

if __name__ == '__main__':
    FLAGS(sys.argv)

    dataset = KITTIObjectDataset(
        root_dir=FLAGS.root_dir,
        rendering_dir=FLAGS.rendering_dir,
        zmax=70.0,
        occlusion_level='fully_visible',
        split=FLAGS.split,
        transforms=None,
        subset_ratio=1.0,
        pose_positive_threshold=1,
        object_subcategory=0,
        device=torch.device('cpu'),
        object_category=FLAGS.category,
        downsample_rate=2,
        labelling_method='euler',
        to_bgr=False,
    )
    c = (0, 0, 255)
    s = 0.6
    t = 1
    idx = FLAGS.start
    h, w = dataset._image_size

    while True:
        try:
            anchor, positive, label = dataset[idx]
            conf = dataset._image_crop_labels[idx]['conf']
            anchor = cv2.resize(anchor, None, fx=2, fy=2)
            positive = cv2.resize(positive, None, fx=2, fy=2)
            img = np.hstack((anchor, positive))
            cv2.putText(img, f'{idx}/{len(dataset)}', (4*w-120, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                        thickness=t, color=c)
            cv2.putText(img, f'[azim, elev, theta] = {label}', (2*w+20, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                        thickness=t, color=c)
            cv2.putText(img, f'confidence = {conf:.2f}', (2*w+20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                        thickness=t, color=c)
            cv2.imshow('Instance', img)

            key = None
            while key not in [27, 81, 83, ord('q')]:
                key = cv2.waitKey(0)

            if key == ord('q'):
                break
            elif key == 81:
                idx = max(0, idx-1)
            elif key == 83:
                idx = min(idx+1, len(dataset))
        except Exception as err:
            traceback.print_exc()
            break

    cv2.destroyAllWindows()