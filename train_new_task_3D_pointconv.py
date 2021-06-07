import os, sys, copy
import torch
import torch.nn.parallel
import numpy as np
import argparse
import scipy.io as sio
import yaml
from tqdm import tqdm
# from src.pointnet import PointNetCls, PointnetLoss
from src.pointconv import PointConvDensityClsSsg, PointConvDensityClsSsgSem
from src.losses import *
from src.models_3D import *
from src.provider import *
from src.data_utils.datautil_3D import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

"""
Arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ModelNet', choices=['ModelNet', 'ScanObjectNN'], help='name of dataset i.e. ModelNet, ScanObjectNN, McGill')
parser.add_argument('--epoch', default=30, type=int, help='number of epochs')
parser.add_argument('--sem', default='w2v', type=str, choices=['w2v', 'glove', 'none'], help='using sem representation ie.e w2v, glove, none')

args = parser.parse_args()

config_file = open('config/'+args.dataset+'_config.yaml', 'r') # args.config_path
config = yaml.load(config_file, Loader=yaml.FullLoader)
config = {**config, **vars(args)} # merger args and config
print(config)


"""
Hyperparameters 
"""
epochs = config['epoch']
batch_size = int(config['batch_size'])
lr =  float(config['lr'])
amsgrad = True
eps = 1e-8
wd = float(config['wd'])
# python train_new_task_3D_pointconv.py --data ScanObjectNN --sem none --epoch 20 

""""
Dataset
"""
trainDataLoaderNew, testDataLoaderOld, testDataLoaderNew, old_w2v, new_w2v, old_att, new_att = get_dataset(config, normal=True)

"""
Model
"""
OLD_MODEL_PATH = config['pointconv_old_model_path_'+config['sem']]
SAVED_MODEL_PATH = config['pointconv_old_model_path_'+config['sem']].replace('old', 'new')
print('Old model path :', OLD_MODEL_PATH)
print('Saved model path :', SAVED_MODEL_PATH)

if config['sem']=='none':
    old_model = PointConvDensityClsSsg(num_classes=config['seen_class'])
else:
    old_model = PointConvDensityClsSsgSem(att_size=old_att.shape[1])
old_model.load_state_dict(torch.load(OLD_MODEL_PATH),strict=False)

for param in old_model.parameters():
    param.requires_grad = False
old_model.to(device)
old_model = old_model.eval()
# print(old_model)
# for name, param in old_model.named_parameters():
#     print(name, param.requires_grad)
# sys.exit()

shared_model = copy.deepcopy(old_model)
if config['sem']!='none':
    # with w2v
    new_model = PointConvLwf3Dv2_2(shared_model, att_size=old_att.shape[1]).to(device) 
else:
    # without w2v
    new_model = PointConvLwf3Dv1_2(shared_model, num_classes=config['unseen_class']).to(device) 

optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps, amsgrad=amsgrad)

# loss
# point_loss = nn.NLLLoss().to(device)
hinton_loss = SoftTargetKDLoss(T=float(config['T'])).to(device)


"""
Training and evaluation
"""
for epoch in range(epochs):
    print('Epoch : ',epoch+1)
    # Training
    t = tqdm(enumerate(trainDataLoaderNew, 0), total=len(trainDataLoaderNew), ncols=100, smoothing=0.9, position=0, leave=True)
    for batch_id, data in t:
        points, target = data
        points = points.data.numpy()
        # print(points.shape)
        jittered_data = random_scale_point_cloud(points[:,:, 0:3], scale_low=2.0/3, scale_high=3/2.0)
        jittered_data = shift_point_cloud(jittered_data, shift_range=0.2)
        points[:, :, 0:3] = jittered_data
        points = random_point_dropout_v2(points)
        shuffle_points(points)
        points = torch.Tensor(points)
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        # old task's target
        # print(points.size())
        old_model.eval()
        new_model.train()
        if config['dataset'] == 'ModelNet':
            if config['sem']=='none':
                old_target, _ = old_model(points[:, :3, :], points[:, 3:, :])
            else:
                old_target, _ = old_model(points[:, :3, :], points[:, 3:, :], old_w2v)
            if config['sem']!='none':
                old_pred, new_pred, new_feat = new_model(points[:, :3, :], points[:, 3:, :], old_w2v, new_w2v) # with w2v
            else:
                old_pred, new_pred, new_feat = new_model(points[:, :3, :], points[:, 3:, :]) # without w2v
        
        if config['dataset'] == 'ScanObjectNN':
            if config['sem']=='none':
                old_target, _ = old_model(points[:, :3, :], points[:, :3, :])
            else:
                old_target, _ = old_model(points[:, :3, :], points[:, :3, :], old_w2v)
            if config['sem']!='none':
                old_pred, new_pred, new_feat = new_model(points[:, :3, :], points[:, :3, :], old_w2v, new_w2v) # with w2v
            else:
                old_pred, new_pred, new_feat = new_model(points[:, :3, :], points[:, :3, :]) # without w2v
        # loss calculation
        loss = F.nll_loss(new_pred, target)      
        kd_loss= hinton_loss(old_pred, old_target)
        t.set_postfix(KDLoss=kd_loss.item(), PointLoss=loss.item())
        loss = loss + 3.0*kd_loss
        loss.backward()
        optimizer.step()
    # sys.exit()

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
            points = points.permute(0, 2, 1)
            points, target = points.to(device).float(), target.to(device).long()
            if config['sem']=='none':
                pred, _ = old_model(points[:, :3, :], points[:, 3:, :])
            else:
                pred, _ = old_model(points[:, :3, :], points[:, 3:, :], old_att)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        instance_acc = np.mean(mean_correct)*100
        print('Accuracy of old task from old model :', round(instance_acc,2))

        # accuracy for old task from new model
        num_class = config['seen_class']
        new_model.eval()
        mean_correct = []
        for j, data in enumerate(testDataLoaderOld):
            points, target = data
            target = target[:, 0]
            points = points.permute(0, 2, 1)
            points, target = points.to(device).float(), target.to(device).long()
            if config['sem']!='none':
                pred, _, _ = new_model(points[:, :3, :], points[:, 3:, :], old_att, new_att) # with w2v
            else:
                pred, _, _ = new_model(points[:, :3, :], points[:, 3:, :]) # without w2v
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        old_acc = np.mean(mean_correct)*100
        print('Accuracy of old task from new model :', round(old_acc,2))
        
        # accuracy for new task from new model
        num_class = config['unseen_class']
        new_model.eval()
        mean_correct = []
        for j, data in enumerate(testDataLoaderNew):
            points, target = data
            target = target[:, 0]
            points = points.permute(0, 2, 1)
            points, target = points.to(device).float(), target.to(device).long()
            if config['dataset'] == 'ModelNet':
                if config['sem']!='none':
                    _, pred, _= new_model(points[:, :3, :], points[:, 3:, :], old_att, new_att) # with w2v
                else:
                    _, pred, _ = new_model(points[:, :3, :], points[:, 3:, :]) # without w2v
            if config['dataset'] == 'ScanObjectNN':
                if config['sem']!='none':
                    _, pred, _ = new_model(points[:, :3, :], points[:, :3, :], old_att, new_att) # with w2v
                else:
                    _, pred, _ = new_model(points[:, :3, :], points[:, :3, :]) # without w2v
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item()/float(points.size()[0]))
        new_acc = np.mean(mean_correct)*100
        print('Accuracy of new task from new model :', round(new_acc,2))
        diff = round((instance_acc-old_acc)*100/instance_acc, 2)
        print('Difference :', diff)

