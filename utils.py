import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import laspy
import torch


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, 'train', 'PointCloud')
# g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'class_names.txt'))]
g_classes = [x.rstrip() for x in open('class_names.txt')]
g_indices = [y.rstrip() for y in open('class_indices.txt')]
g_colors = [[255, 255, 255],
            [0, 0, 255],
            [20, 150, 20],
            [255, 0, 0],
            [255, 255, 0]]

g_class2label = dict(zip(g_classes, g_indices))
g_class2color = dict(zip(g_classes, g_colors))


