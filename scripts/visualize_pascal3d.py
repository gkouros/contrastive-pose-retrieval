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

# local imports
sys.path.append(sys.path[0] + '/../src')
from dataloader.pascal3dplus_dataset import PASCAL3DPlusDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', '/users/visics/gkouros/projects/datasets/pascal3d',
                    'The location of the dataset')
flags.DEFINE_boolean('symmetric', False, 'Displays the flipped image as well')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'test'], 'Dataset split')
flags.DEFINE_enum('category', 'car', ['car'], 'Object category')
flags.DEFINE_enum('positive', 'rendering',
                  ['rendering', 'silhouette', 'depth', 'normals', 'all'],
                  'Positive type')
flags.DEFINE_integer('start', 0, 'Starting idx to iterate from')
flags.DEFINE_boolean('positive_from_db', False, 'Set to True to load '
                     'positives from the large database')

if __name__ == '__main__':
    FLAGS(sys.argv)

    dataset = PASCAL3DPlusDataset(
        root_dir=FLAGS.root_dir,
        split=FLAGS.split,
        transforms=None,
        subset_ratio=1.0,
        positive_type=FLAGS.positive,
        horizontal_flip=True,
        object_category=FLAGS.category,
        render_from_scratch=False,
        data_from_scratch=False,
        positive_from_db=FLAGS.positive_from_db,
        downsample_rate=1,
        labelling_method='euler',
        to_bgr=True,
    )
    c = (0, 0, 255)
    s = 0.6
    t = 2
    idx = FLAGS.start
    idx2 = idx + len(dataset) // 2
    h, w = dataset._image_size

    while True:
        try:
            anchor, positive, label = dataset[idx]
            img = np.hstack((anchor, positive))
            cv2.putText(img, f'{idx}/{len(dataset)}', (2*w-120, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                        thickness=t, color=c)
            cv2.putText(img, f'[azim, elev, theta] = {label}', (w+20, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                        thickness=t, color=c)

            if FLAGS.symmetric:
                anchor2, positive2, label2 = dataset[idx2]
                img2 = np.hstack((anchor2, positive2))
                cv2.putText(img2, f'{idx2}/{len(dataset)}', (2*w-120, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                            thickness=t, color=c)
                cv2.putText(img2, f'[azim, elev, theta] = {label2}', (w+20, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=s,
                            thickness=t, color=c)
                cv2.imshow('Instance', np.vstack((img, img2)))
            else:
                cv2.imshow('Instance', img)

            key = None
            while key not in [27, 81, 83, ord('q')]:
                key = cv2.waitKey(0)

            if key == ord('q'):
                break
            elif key == 81:
                idx = max(0, idx-1)
                idx2 = max(0, idx2-1)
            elif key == 83:
                idx = min(idx+1, len(dataset))
                idx2 = min(idx2+1, len(dataset))
        except Exception as err:
            traceback.print_exc()
            break

    cv2.destroyAllWindows()