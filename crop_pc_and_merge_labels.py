import numpy as np
import laspy as lp
import os
import matplotlib.pyplot as plt
from math import ceil

def convert_color(r16, g16, b16):

    r8 = ceil((float(r16/65535)) * float(255))
    g8 = ceil((float(g16 / 65535)) * float(255))
    b8 = ceil((float(b16 / 65535)) * float(255))

    return [r8, g8, b8]


def calculate_weights(labels, classIndices):
    weights = np.zeros(len(classIndices))
    for i in range (0, len(classIndices)):
        sum = 0
        for j in range(0, len(labels)):
            if labels[j] == classIndices[i]:
                sum = sum + 1
        weights[i] = sum

    return weights

def crop_and_merge(gridSize, datasetPath,trainsetPath, numPoints, writeFiles, numClasses, classIndices):

    weights = np.ones(numClasses)
    fileNames = os.listdir(trainsetPath)
    ptCloudPath = os.path.join(datasetPath, 'PointCloud')
    labelsPath = os.path.join(datasetPath, 'Labels')
    num = 1

    if writeFiles:
        weights = np.zeros(numClasses)

        for j in range (0, len(fileNames)):
            if len(fileNames[j].split('.')) > 1:
                if fileNames[j].split('.')[1] != 'las':
                    break

            filePath = os.path.join(trainsetPath, fileNames[j])
            pc = lp.read(filePath)
            pc.X = pc.X / 1000
            pc.Y = pc.Y / 1000
            pc.Z = pc.Z / 1000

            labels = pc.classification
            weights = weights + calculate_weights(labels, classIndices)

            pc_array = np.zeros((labels.shape[0], 7))
            for i in range(0, pc_array.shape[0]):
                [r, g, b] = convert_color(pc.red[i], pc.green[i], pc.blue[i])
                pc_list = [pc.X[i], pc.Y[i], pc.Z[i], r, g, b, labels[i]]
                pc_array[i] = pc_list

            np.save(os.path.join(ptCloudPath, 'PC_' + fileNames[j].split('.')[0]), pc_array)
            num = num + 1

            # xLimits =[min(pc.X), max(pc.X)]
            # yLimits = [min(pc.Y), max(pc.Y)]
            # zLimits = [min(pc.Z), max(pc.Y)]
            #
            # numGridsX = round((xLimits[1]- xLimits[0])/ gridSize[0])
            # numGridsY = round((yLimits[1] - yLimits[0]) / gridSize[1])
            #
            # N, xEdges, yEdges = np.histogram2d(pc.X, pc.Y, bins=[numGridsX, numGridsY])
            # xbins = xEdges[0:-1]
            # ybins= yEdges[0:-1]
            # indx = np.digitize(pc.X, xbins)
            # indy = np.digitize(pc.Y, ybins)
            #
            # ind_init = np.vstack((indx, indy))
            # ind = np.ravel_multi_index(ind_init, (numGridsX+1, numGridsY+1))

            # for k in range(0, numGridsX*numGridsY):
            #     idx = ind == k
            #     ptCloudDense = pc[idx]
            #     labelsDense = labels[idx]
            #     if labelsDense.shape[0] > 0:
            #         pc_array = np.zeros((labelsDense.shape[0], 7))
            #
            #         for i in range(0, pc_array.shape[0]):
            #             [r, g, b] = convert_color(ptCloudDense.red[i], ptCloudDense.green[i], ptCloudDense.blue[i])
            #             pc_list = [ptCloudDense.X[i], ptCloudDense.Y[i], ptCloudDense.Z[i], r, g, b, labelsDense[i]]
            #             pc_array[i] = pc_list
            #
            #         np.save(os.path.join(ptCloudPath, 'PC_' + fileNames[j].split('.')[0] + '_' + str(num)), pc_array)
            #         num = num + 1

    return [ptCloudPath, labelsPath, weights]