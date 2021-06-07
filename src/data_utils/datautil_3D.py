import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from src.data_utils.modelnet_data_util import ModelNetDataset
from src.data_utils.scanobjectnn_data_util import ScanObjectNNDataset


def get_dataset(config, normal=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = int(config['batch_size'])
    
    OLD_DATA_PATH = 'D:/NSU/sfr1/Journal/journal_code/modelnet40_normal_resampled'
    NEW_DATA_PATH = 'D:/NSU/sfr1/BMVC/dataset/scanobjectnn'
    
    if config['sem']=='w2v':
        wordvector = sio.loadmat(os.path.join(OLD_DATA_PATH, 'word_vector/ModelNet40_w2v'))
        w2v = wordvector['word']
    
    if config['sem']=='glove':
        wordvector = sio.loadmat(os.path.join(OLD_DATA_PATH, 'word_vector/ModelNet40_glove'))
        w2v = wordvector['word']
    
    trainDataLoaderNew, testDataLoaderOld, testDataLoaderNew, old_w2v, new_w2v, old_att, new_att = None, None, None, None, None, None, None

    if config['dataset'] == 'ScanObjectNN':
        TRAIN_DATASET_NEW = ScanObjectNNDataset(root=NEW_DATA_PATH, npoint=1024, split='train')
        TEST_DATASET_OLD = ModelNetDataset(root=OLD_DATA_PATH, settings='basic_26', npoint=1024, split='test', normal_channel=normal)
        TEST_DATASET_NEW = ScanObjectNNDataset(root=NEW_DATA_PATH, npoint=1024, split='test')

        trainDataLoaderNew = torch.utils.data.DataLoader(TRAIN_DATASET_NEW, batch_size=batch_size, shuffle=True, num_workers=0)
        testDataLoaderOld = torch.utils.data.DataLoader(TEST_DATASET_OLD, batch_size=batch_size, shuffle=False, num_workers=0)
        testDataLoaderNew = torch.utils.data.DataLoader(TEST_DATASET_NEW, batch_size=batch_size, shuffle=False, num_workers=0)

        # semantic representation
        if config['sem']!='none':
            seen_set_index = np.int16([0,1,5,6,7,9,10,11,15,16,17,18,19,20,21,23,24,25,26,27,28,31,34,36,37,39])
            unseen_set_index = np.int16([12,22,13,4,33,2,29,30,35])

            v1 = (w2v[14,:]+w2v[38,:])/2
            v2 =  (w2v[8,:]+w2v[32,:]+w2v[3,:])/3
            v1 = np.expand_dims(v1, 0)
            v2 = np.expand_dims(v2, 0)

            old_w2v = torch.tensor(w2v[seen_set_index, :], requires_grad=True).float().to(device)
            new_w2v = torch.tensor(np.concatenate((v1, v2, w2v[unseen_set_index, :]), axis=0), requires_grad=True).float().to(device)
            old_att = torch.from_numpy(w2v[seen_set_index, :]).float().to(device)
            new_att = torch.from_numpy(np.concatenate((v1, v2, w2v[unseen_set_index, :]), axis=0)).float().to(device)

    elif config['dataset'] == 'ModelNet':
        # TRAIN_DATASET_OLD = ModelNetDataset(root=DATA_PATH, settings='basic_30', npoint=1024, split='train', normal_channel=False)
        TRAIN_DATASET_NEW = ModelNetDataset(root=OLD_DATA_PATH, settings='basic_10', npoint=1024, split='train', normal_channel=normal)
        TEST_DATASET_OLD = ModelNetDataset(root=OLD_DATA_PATH, settings='basic_30', npoint=1024, split='test', normal_channel=normal)
        TEST_DATASET_NEW = ModelNetDataset(root=OLD_DATA_PATH, settings='basic_10', npoint=1024, split='test', normal_channel=normal)

        # trainDataLoaderOld = torch.utils.data.DataLoader(TRAIN_DATASET_OLD, batch_size=batch_size, shuffle=True, num_workers=0)
        trainDataLoaderNew = torch.utils.data.DataLoader(TRAIN_DATASET_NEW, batch_size=batch_size, shuffle=True, num_workers=0)
        testDataLoaderOld = torch.utils.data.DataLoader(TEST_DATASET_OLD, batch_size=batch_size, shuffle=False, num_workers=0)
        testDataLoaderNew = torch.utils.data.DataLoader(TEST_DATASET_NEW, batch_size=batch_size, shuffle=False, num_workers=0)

        # semantic representation
        if config['sem']!='none':
            seen_set_index = np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
            unseen_set_index =np.int16([1,2,8,12,14,22,23,30,33,35])
            
            old_w2v = torch.tensor(w2v[seen_set_index, :], requires_grad=True).float().to(device)
            new_w2v = torch.tensor(w2v[unseen_set_index, :], requires_grad=True).float().to(device)
            old_att = torch.from_numpy(w2v[seen_set_index, :]).float().to(device)
            new_att = torch.from_numpy(w2v[unseen_set_index, :]).float().to(device)
    
    return trainDataLoaderNew, testDataLoaderOld, testDataLoaderNew, old_w2v, new_w2v, old_att, new_att
