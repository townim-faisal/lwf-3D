import numpy as np
import warnings
import os, sys
import torch
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import scipy.io as sio


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


"""
Input:
    xyz: pointcloud data, [N, D]
    npoint: number of samples
Return:
    centroids: sampled pointcloud index, [npoint, D]
"""
def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset(Dataset):
    def __init__(self, root, settings, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        assert (split == 'train' or split == 'test')
        assert (settings == 'basic_40' or settings == 'basic_30' or settings == 'basic_10' or settings == 'basic_26')
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        
        if settings=='basic_40':
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        if settings == 'basic_10':
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        if settings=='basic_30':
            self.catfile = os.path.join(self.root, 'ModelNet_class_names_seen.txt')
        if settings=='basic_26':
            self.catfile = os.path.join(self.root, 'modelnet26_shape_names.txt')
        # if (settings=='w2v' or settings=='glove') and split=='train':
        #     self.catfile = os.path.join(self.root, 'ModelNet_class_names_seen.txt')
        # if (settings=='w2v' or settings=='glove') and split=='test':
        #     self.catfile = os.path.join(self.root, 'ModelNet_class_names_unseen.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_names = []
        shape_ids[split] = []
        
        if split=='train': # or (settings=='semantics' and split=='test'):
            for line in open(os.path.join(self.root, 'modelnet40_train.txt')):
                shape = '_'.join(line.rstrip().split('_')[0:-1])
                if shape in self.classes:
                    shape_names.append(shape)
                    shape_ids[split].append(line.rstrip())
        
        if split=='test':
            for line in open(os.path.join(self.root, 'modelnet40_test.txt')):
                shape = '_'.join(line.rstrip().split('_')[0:-1])
                if shape in self.classes:
                    shape_names.append(shape)
                    shape_ids[split].append(line.rstrip())
        
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of '+split+' data for '+settings+' is '+str(len(self.datapath))+' and no of classes is '+str(len(self.cat)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        # seen_set_index = np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
        # unseen_set_index =np.int16([1,2,8,12,14,22,23,30,33,35])
        # wordvector = sio.loadmat(os.path.join(self.root, 'word_vector/ModelNet40_w2v'))
        # w2v = wordvector['word']
        # if settings=='basic_30':
        #     self.semantics = w2v[seen_set_index,:]
        # if settings=='basic_10':
        #     self.semantics = w2v[unseen_set_index,:]

        del shape_ids, shape_names

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            # sem = np.array(self.semantics[cls])
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)





if __name__ == '__main__':
    data = ModelNetDataset(root='D:/NSU/sfr1/Journal/journal_code/modelnet40_normal_resampled', settings='basic_30', split='train', uniform=False, normal_channel=True,)
    print(data.classes)
    print(data.datapath)
    sys.exit()
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label)
        break