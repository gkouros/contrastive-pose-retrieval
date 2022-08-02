import numpy as np


AZIMUTH_OFFSET = 90

OBJECT_CATEGORIES = []
for cat in ['car', 'bus', 'motorbike', 'bottle', 'boat', 'bicycle',
            'aeroplane', 'sofa', 'tvmonitor', 'chair', 'diningtable', 'train']:
    for level in range(0, 4):
        OBJECT_CATEGORIES.append(cat + (level > 0) * f'FGL{level}_BGL{level}')

IMAGE_SIZES = {
    'car': (256, 672), 'bus': (320, 800), 'motorbike': (512, 512),
    'boat': (480, 1120), 'bicycle': (608, 608), 'aeroplane': (320, 1024),
    'sofa': (352, 736), 'tvmonitor': (480, 480), 'chair': (544, 384),
    'diningtable': (320, 800), 'bottle': (512, 736), 'train': (256, 608)}

CATEGORY_DISTANCES = {
    'car': 5, 'bus': 5.2, 'motorbike': 4.5, 'bottle': 5.75, 'boat': 8,
    'bicycle': 5.2, 'aeroplane': 7, 'sofa': 5.4, 'tvmonitor': 5.5, 'chair': 4,
    'diningtable': 7, 'train': 3.75}

ANGLE_DISCRETIZATIONS = {'azimuth': 12, 'elevation': 4, 'theta': 3}

ANGLE_RANGES = {
    'azimuth': np.linspace(0, 360, ANGLE_DISCRETIZATIONS['azimuth'],
                           endpoint=False),
    'elevation': np.linspace(-30, 60, ANGLE_DISCRETIZATIONS['elevation']),
    'theta': np.linspace(-30, 30, ANGLE_DISCRETIZATIONS['theta'])}

ANGLE_STEPS = {'azimuth': 30, 'elevation': 30, 'theta': 30}

ANGLE_LABELS = {}
label = 0
for azim in range(ANGLE_DISCRETIZATIONS['azimuth']):
    for elev in range(ANGLE_DISCRETIZATIONS['elevation']):
        for theta in range(ANGLE_DISCRETIZATIONS['theta']):
            ANGLE_LABELS[(azim, elev, theta)] = label
            label += 1

RENDERING_FORMATS = {
    'rendering': 'JPEG',
    'silhouette': 'JPEG',
    'depth': 'TIFF',
    'normals': 'TIFF',
}