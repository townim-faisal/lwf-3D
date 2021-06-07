import os, sys, copy
import torch
import torch.nn.parallel
import numpy as np
import argparse
import scipy.io as sio
import yaml
from tqdm import tqdm
from src.pointnet import PointNetCls, PointNetClsSem, PointnetLoss
# from src.pointconv import PointConvDensityClsSsg
from src.losses import *
from src.models_3D import *
from src.provider import *
from src.data_utils.datautil_3D import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
parser.add_argument('--epoch', default=60, type=int, help='number of epochs')
parser.add_argument('--sem', default='w2v', type=str, choices=['w2v', 'glove', 'none'], help='using sem representation ie.e w2v, glove, none')


args = parser.parse_args()
config_file = open('config/'+args.dataset+'_config.yaml', 'r') # args.config_path
config = yaml.load(config_file, Loader=yaml.FullLoader)
config = {**config, **vars(args)} # merger args and config

"""
Hyperparameters 
"""
epochs = config['epoch']
batch_size = int(config['batch_size'])
lr =  float(config['lr'])
amsgrad = True
eps = 1e-8
wd = float(config['wd'])
# config['T'] = 5
print(config)

""""
Dataset
"""
trainDataLoaderNew, testDataLoaderOld, testDataLoaderNew, old_w2v, new_w2v, old_att, new_att = get_dataset(config)

"""
Model
"""
OLD_MODEL_PATH = config['pointnet_old_model_path_'+config['sem']]
SAVED_MODEL_PATH = config['pointnet_old_model_path_'+config['sem']].replace('old', 'new')
print('Old model path :', OLD_MODEL_PATH)
print('Saved model path :', SAVED_MODEL_PATH)

if config['sem']=='none':
    old_model = PointNetCls(k=config['seen_class'], feature_transform=True)
else:
    old_model = PointNetClsSem(feature_transform=True, att_size=old_att.shape[1])
old_model.load_state_dict(torch.load(OLD_MODEL_PATH), strict=True)
old_model.to(device)
old_model = old_model.eval()

for param in old_model.parameters():
    param.requires_grad = False
# print(old_model)
# sys.exit()

shared_model = copy.deepcopy(old_model)
if config['sem'] != 'none':
    # with w2v
    new_model = PointNetLwf3Dv2_2(shared_model, att_size=old_att.shape[1]).to(device) 
else:
    # without w2v
    new_model = PointNetLwf3Dv1_2(shared_model, config).to(device) 

optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps, amsgrad=amsgrad)

# loss
point_loss = PointnetLoss(mat_diff_loss_scale=0.001).to(device)
hinton_loss = SoftTargetKDLoss(T=float(config['T'])).to(device)

"""
Training and evaluation
"""
lamb = 3
for epoch in range(epochs):
    print('Epoch : ',epoch+1)
    # Training
    t = tqdm(enumerate(trainDataLoaderNew, 0), total=len(trainDataLoaderNew), ncols=100, smoothing=0.9, position=0, leave=True)
    for batch_id, data in t:
        points, target = data
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:,:, 0:3] = random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device).float(), target.to(device)
        optimizer.zero_grad()
        # old task's target
        old_model.eval()
        if config['sem']=='none':
            old_target, old_feat, _ = old_model(points)
        else:
            old_target, old_feat, _ = old_model(points, old_w2v)
        # new model's training
        new_model.train()
        if config['sem']!='none':
            old_pred, new_pred, new_feat, trans_feat = new_model(points, old_w2v, new_w2v) # with w2v
        else:
            old_pred, new_pred, new_feat, trans_feat = new_model(points) # without w2v
        # loss calculation
        loss = point_loss(new_pred, target.long(), trans_feat)
        kd_loss= hinton_loss(old_pred, old_target)
        t.set_postfix(KDLoss=kd_loss.item(), PointLoss=loss.item())
        loss = loss + lamb*kd_loss
        loss.backward()
        optimizer.step()

    # scheduler.step()

    # Evaluation
    with torch.no_grad():
        # accuracy for old task from old model
        num_class=config['seen_class']
        old_model.eval()
        mean_correct = []
        for j, data in enumerate(testDataLoaderOld):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device).float(), target.to(device)
            if config['sem']=='none':
                pred, _, _ = old_model(points)
            else:
                pred, _, _ = old_model(points, old_att)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        old_lwf_acc = round(np.mean(mean_correct)*100, 2)
        print('Accuracy of old task from old model :', old_lwf_acc)
        
        # accuracy for old task from new model
        num_class = config['seen_class']
        new_model.eval()
        mean_correct = []
        for j, data in enumerate(testDataLoaderOld):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            if config['sem']!='none':
                pred, _, _, _ = new_model(points.float(), old_att, new_att) # with w2v
            else:
                pred, _, _, _ = new_model(points.float()) # without w2v
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        old_acc = round(np.mean(mean_correct)*100, 2)
        print('Accuracy of old task from new model :', old_acc)
        
        # accuracy for new task from new model
        num_class = config['unseen_class']
        new_model.eval()
        mean_correct = []
        for j, data in enumerate(testDataLoaderNew):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            if config['sem']!='none':
                _, pred, _, _ = new_model(points.float(), old_att, new_att) # with w2v
            else:
                _, pred, _, _ = new_model(points.float()) # without w2v
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        new_acc = round(np.mean(mean_correct)*100, 2)
        print('Accuracy of new task from new model :', new_acc)
        diff = round((old_lwf_acc-old_acc)*100/old_lwf_acc, 2)
        print('Difference :', diff)

       

