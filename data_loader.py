import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class DataLoader(Dataset):

    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        areas = sorted(os.listdir(data_root))
        areas = [area for area in areas if 'PC_' in area]
        if split == 'train':
            areas_split = [area for area in areas if not 'PC_{}'.format(test_area) in area]
        else:
            areas_split = [area for area in areas if 'PC_{}'.format(test_area) in area]

        self.area_points, self.area_labels = [], []
        self.area_coord_min, self.area_coord_max = [], []
        self.labelweights = np.zeros(9)
        num_point_all = []
        labelweights_init = np.zeros(9)
        labelweights = np.zeros(9)

        for area_name in tqdm(areas_split, total=len(areas_split)):
            area_path = os.path.join(data_root, area_name)
            area_data = np.load(area_path)  # xyzrgbl, N*7
            points, labels = area_data[:, 0:6], area_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(10)) #11 IF WE HAVE ALSO WATER
            print("TMP ", tmp)
            labelweights_init += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.area_points.append(points), self.area_labels.append(labels)
            self.area_coord_min.append(coord_min), self.area_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        print("Initially: ", labelweights_init)
        print(np.sum(labelweights_init))
        labelweights_init = labelweights_init.astype(np.float32)
        labelweights = labelweights.astype(np.float32)

        for w in range (0, labelweights.shape[0]):
          if labelweights_init[w] == 0:
            labelweights[w] = labelweights_init[w]
          else:
            labelweights[w] = labelweights_init[w] / np.sum(labelweights_init)
        # labelweights = labelweights[labelweights!=0] / np.sum(labelweights)
        for x in range (0, labelweights.shape[0]):
          if labelweights[x] == 0:
            self.labelweights[x] = 0
          else:
            self.labelweights[x] = np.power(np.amax(labelweights) / labelweights[x], 1 / 3.0)
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        area_idxs = []
        for index in range(len(areas_split)):
            area_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.area_idxs = np.array(area_idxs)
        print("Totally {} samples in {} set.".format(len(self.area_idxs), split))

    def __getitem__(self, idx):
        area_idx = self.area_idxs[idx]
        points = self.area_points[area_idx]  # N * 6
        labels = self.area_labels[area_idx]  # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                            points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.area_coord_max[area_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.area_coord_max[area_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.area_coord_max[area_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.area_idxs)
