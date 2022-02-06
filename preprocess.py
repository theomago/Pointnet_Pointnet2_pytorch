import numpy as np
import laspy as lp
import os
import sys
import time
import matplotlib.pyplot as plt
# import torch

from crop_pc_and_merge_labels import *

np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True)

start = time.time()

DATA_FOLDER = os.path.join('C:/Users/Theodora/Desktop/', 'C_69AZ1')
TRAIN_DATA_FOLDER = os.path.join(DATA_FOLDER, 'train')

# pc = lp.read(os.path.join(TRAIN_DATA_FOLDER, '1.las'))

classNames = ['unclassified',
              'ground',
              'vegetation',
              'building',
              'water']

classIndices = [1,
                2,
                3,
                6,
                9]

numClasses = len(classNames)
gridSize = [50,50]
numPoints = 8192
writeFiles = True

weights = np.ones(numClasses)
fileNames = os.listdir(TRAIN_DATA_FOLDER)
ptCloudPath = os.path.join(DATA_FOLDER, 'PointCloud2')
labelsPath = os.path.join(DATA_FOLDER, 'Labels')
num = 1

p1, p2, w = crop_and_merge(gridSize, DATA_FOLDER, TRAIN_DATA_FOLDER, numPoints, writeFiles, numClasses, classIndices)

end = time.time()
print(f"Runtime of the program is {end - start}")