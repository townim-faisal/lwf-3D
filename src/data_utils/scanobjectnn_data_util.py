import numpy as np
import warnings
import os
import h5py
import torch
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import scipy.io as sio


class ScanObjectNNDataset(Dataset):
    def __init__(self, root, npoint=1024, split='train'):
        assert (split == 'train' or split == 'test')
        
        if split=='test':
            f = h5py.File(os.path.join(root,'test_objectdataset.h5'))
        if split=='train':
            f = h5py.File(os.path.join(root,'training_objectdataset.h5'))
        data = f['data'][:]
        label = f['label'][:]
        label_selected = [3,4,5,6,7,8,9,10,12,13,14]

        freq = np.zeros(len(label_selected))
        for i in range(len(label_selected)):
            mark = np.where(label==label_selected[i] )
            freq[i] = len(mark[0])
        

        self.data_n = np.zeros((int(np.sum(freq)), npoint, 3))
        self.label_n = np.zeros(int(np.sum(freq)))

        c = 0
        for i in range(11):
            idx = np.where(label==label_selected[i])
            for j in range(idx[0].shape[0]):
                self.label_n[c] = i
                self.data_n[c,:,:] = data[idx[0][j],0:npoint,:]
                c = c + 1
        print( 'The size of '+split+' data for ScanObjectNN is '+str(len(self.label_n))+' and no of classes is '+str(len(np.unique(self.label_n))) )

    def __len__(self):
        return len(self.label_n)

    def __getitem__(self, index):
        return self.data_n[index,:,:], np.array([self.label_n[index]]).astype(np.int32)





if __name__ == '__main__':
    data = ScanObjectNNDataset(root='D:/NSU/sfr1/BMVC/dataset/scanobjectnn', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label)
        break