import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.dgcnn import knn, get_graph_feature

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            m.bias.data.fill_(0)

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')

"""
lwf model for 3D data - PointNet
- as usual LwF model - task speicific layers increased
"""
class PointNetLwf3Dv1_2(nn.Module):
    def __init__(self, shared_model, config):
        super(PointNetLwf3Dv1_2, self).__init__()
        for param in shared_model.parameters():
            param.requires_grad = True
        fc1 = shared_model.fc1
        fc2 = shared_model.fc2
        fc3 = shared_model.fc3
        dropout = shared_model.dropout
        bn1 = shared_model.bn1
        bn2 = shared_model.bn2
        self.shared_model = shared_model.feat

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'fc1': fc1,
                'bn1': bn1,
                'dropout': dropout,
                'fc2': fc2,
                'bn2': bn2,
                'fc3': fc3
            }),
            nn.ModuleDict({
                'fc1': nn.Linear(fc1.in_features, 512),
                'bn1': nn.BatchNorm1d(512),
                'dropout': nn.Dropout(p=0.3),
                'fc2': nn.Linear(512, 256),
                'bn2': nn.BatchNorm1d(256),
                'fc3': nn.Linear(256, config['unseen_class'])
            })
        ])

        self.classifiers[1].apply(init_weights)

    def forward(self, x):
        x, trans, trans_feat = self.shared_model(x)

        # old
        old = F.relu(self.classifiers[0].bn1(self.classifiers[0].fc1(x)))
        old = F.relu(self.classifiers[0].bn2(self.classifiers[0].dropout(self.classifiers[0].fc2(old))))
        feat = old
        old = self.classifiers[0].fc3(old)

        # new
        new = F.relu(self.classifiers[1].bn1(self.classifiers[1].fc1(x)))
        new = F.relu(self.classifiers[1].bn2(self.classifiers[1].dropout(self.classifiers[1].fc2(new))))
        new = self.classifiers[1].fc3(new)
        return F.log_softmax(old, dim=-1), F.log_softmax(new, dim=-1), feat, trans_feat


"""
lwf model for 3D data - PointNet
- using matrix multiplication with w2v's transformation
"""
class PointNetLwf3Dv2_2(nn.Module):
    def __init__(self, shared_model, att_size=300):
        super(PointNetLwf3Dv2_2, self).__init__()
        for param in shared_model.parameters():
            param.requires_grad = True
        fc1 = shared_model.fc1
        fc2 = shared_model.fc2
        # fc3 = self.shared_model.fc3
        dropout = shared_model.dropout
        bn1 = shared_model.bn1
        bn2 = shared_model.bn2
        self.shared_model = shared_model.feat

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'fc1': fc1,
                'bn1': bn1,
                'dropout': dropout,
                'fc2': fc2,
                'bn2': bn2,
                # 'fc3': fc3
            }),
            nn.ModuleDict({
                'fc1': nn.Linear(fc1.in_features, 512),
                'bn1': nn.BatchNorm1d(512),
                'dropout': nn.Dropout(p=0.3),
                'fc2': nn.Linear(512, 256),
                'bn2': nn.BatchNorm1d(256),
                # 'fc3': nn.Linear(256, new_class)
            })
        ])

        self.old_att = shared_model.old_att
        # nn.Sequential(
        #     nn.Linear(att_size, 256),
        #     # nn.Tanh(),
        #     nn.ReLU(), 
        # )

        self.new_att = nn.Sequential(
            nn.Linear(att_size, 256),
            # nn.Tanh(),
            nn.ReLU(), 
        )
        # self.old_att.apply(init_weights)
        self.new_att.apply(init_weights)
        self.classifiers[1].apply(init_weights)

    def forward(self, x, old_att=None, new_att=None):
        x, trans, trans_feat = self.shared_model(x)

        # old
        old = F.relu(self.classifiers[0].bn1(self.classifiers[0].fc1(x)))
        old = F.relu(self.classifiers[0].bn2(self.classifiers[0].dropout(self.classifiers[0].fc2(old))))
        feat = old
        old = F.linear(old, self.old_att(old_att))
        
        # new
        new = F.relu(self.classifiers[1].bn1(self.classifiers[1].fc1(x)))
        new = F.relu(self.classifiers[1].bn2(self.classifiers[1].dropout(self.classifiers[1].fc2(new))))
        new = F.linear(new, self.new_att(new_att))

        return F.log_softmax(old, dim=-1), F.log_softmax(new, dim=-1), feat, trans_feat

"""
lwf model for 3D data - PointConv
- as usual LwF model - task speicific layers increased
"""
class PointConvLwf3Dv1_2(nn.Module):
    def __init__(self, shared_model, num_classes=10):
        super(PointConvLwf3Dv1_2, self).__init__()
        for param in shared_model.parameters():
            param.requires_grad = True
        self.sa1 = shared_model.sa1
        self.sa2 = shared_model.sa2
        self.sa3 = shared_model.sa3
        fc1 = shared_model.fc1
        bn1 = shared_model.bn1
        drop1 = shared_model.drop1
        fc2 = shared_model.fc2
        bn2 = shared_model.bn2
        drop2 = shared_model.drop2
        fc3 = shared_model.fc3

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'fc1': fc1,
                'bn1': bn1,
                'drop1': drop1,
                'fc2': fc2,
                'bn2': bn2,
                'drop2': drop2,
                'fc3': fc3
            }),
            nn.ModuleDict({
                'fc1': nn.Linear(fc1.in_features, 512),
                'bn1': nn.BatchNorm1d(512),
                'drop1': nn.Dropout(0.7),
                'fc2': nn.Linear(512, 256),
                'bn2': nn.BatchNorm1d(256),
                'drop2': nn.Dropout(0.7),
                'fc3': nn.Linear(256, num_classes)
            })
        ])

        self.classifiers[1].apply(init_weights)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        feat = x

        # old
        old = self.classifiers[0].drop1(F.relu(self.classifiers[0].bn1(self.classifiers[0].fc1(x))))
        old = self.classifiers[0].drop2(F.relu(self.classifiers[0].bn2(self.classifiers[0].fc2(old))))
        old = self.classifiers[0].fc3(old)

        # new
        new = self.classifiers[1].drop1(F.relu(self.classifiers[1].bn1(self.classifiers[1].fc1(x))))
        new = self.classifiers[1].drop2(F.relu(self.classifiers[1].bn2(self.classifiers[1].fc2(new))))
        new = self.classifiers[1].fc3(new)
        return F.log_softmax(old, dim=-1), F.log_softmax(new, dim=-1), feat


"""
lwf model for 3D data - PointConv
- using matrix multiplication with w2v's transformation
"""
class PointConvLwf3Dv2_2(nn.Module):
    def __init__(self, shared_model, att_size=300):
        super(PointConvLwf3Dv2_2, self).__init__()
        for param in shared_model.parameters():
            param.requires_grad = True
        self.sa1 = shared_model.sa1
        self.sa2 = shared_model.sa2
        self.sa3 = shared_model.sa3
        fc1 = shared_model.fc1
        bn1 = shared_model.bn1
        drop1 = shared_model.drop1
        fc2 = shared_model.fc2
        bn2 = shared_model.bn2
        drop2 = shared_model.drop2
        # fc3 = shared_model.fc3

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'fc1': fc1,
                'bn1': bn1,
                'drop1': drop1,
                'fc2': fc2,
                'bn2': bn2,
                'drop2': drop2,
                # 'fc3': fc3
            }),
            nn.ModuleDict({
                'fc1': nn.Linear(fc1.in_features, 512),
                'bn1': nn.BatchNorm1d(512),
                'drop1': nn.Dropout(0.7),
                'fc2': nn.Linear(512, 256),
                'bn2': nn.BatchNorm1d(256),
                'drop2': nn.Dropout(0.7),
                # 'fc3': nn.Linear(256, config['unseen_class'])
            })
        ])

        self.old_att = shared_model.old_att
        # nn.Sequential(
        #     nn.Linear(att_size, 256),
        #     # nn.Tanh(),
        #     nn.ReLU(), 
        # )

        self.new_att = nn.Sequential(
            nn.Linear(att_size, 256),
            # nn.Tanh(),
            nn.ReLU(), 
        )
        # self.old_att.apply(init_weights)
        self.new_att.apply(init_weights)
        self.classifiers[1].apply(init_weights)

    def forward(self, xyz, feat, old_att=None, new_att=None):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        feat = x

        # old
        old = self.classifiers[0].drop1(F.relu(self.classifiers[0].bn1(self.classifiers[0].fc1(x))))
        old = self.classifiers[0].drop2(F.relu(self.classifiers[0].bn2(self.classifiers[0].fc2(old))))
        old = F.linear(old, self.old_att(old_att))
        
        # new
        new = self.classifiers[1].drop1(F.relu(self.classifiers[1].bn1(self.classifiers[1].fc1(x))))
        new = self.classifiers[1].drop2(F.relu(self.classifiers[1].bn2(self.classifiers[1].fc2(new))))
        new = F.linear(new, self.new_att(new_att))

        return F.log_softmax(old, dim=-1), F.log_softmax(new, dim=-1), feat


"""
lwf model for 3D data - DGCNN
- as usual LwF model - task speicific layers increased
"""
class DGCNNLwf3Dv1_2(nn.Module):
    def __init__(self, shared_model, k=20, dropout=0.5, num_classes=10):
        super(DGCNNLwf3Dv1_2, self).__init__()
        self.k = k
        for param in shared_model.parameters():
            param.requires_grad = True
        self.bn1 = shared_model.bn1
        self.bn2 = shared_model.bn2
        self.bn3 = shared_model.bn3
        self.bn4 = shared_model.bn4
        self.bn5 = shared_model.bn5
        self.conv1 = shared_model.conv1
        self.conv2 = shared_model.conv2
        self.conv3 = shared_model.conv3
        self.conv4 = shared_model.conv4
        self.conv5 = shared_model.conv5
        linear1 = shared_model.linear1
        bn6 = shared_model.bn6
        dp1 = shared_model.dp1
        linear2 = shared_model.linear2
        bn7 = shared_model.bn7
        dp2 = shared_model.dp2
        linear3 = shared_model.linear3

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'linear1': linear1,
                'bn6': bn6,
                'dp1': dp1,
                'linear2': linear2,
                'bn7': bn7,
                'dp2': dp2,
                'linear3': linear3
            }),
            nn.ModuleDict({
                'linear1': nn.Linear(linear1.in_features, 512, bias=False),
                'bn6': nn.BatchNorm1d(512),
                'dp1': nn.Dropout(p=dropout),
                'linear2': nn.Linear(512, 256),
                'bn7': nn.BatchNorm1d(256),
                'dp2': nn.Dropout(p=dropout),
                'linear3': nn.Linear(256, num_classes)
            })
        ])

        self.classifiers[1].apply(init_weights)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # old
        old = F.leaky_relu(self.classifiers[0].bn6(self.classifiers[0].linear1(x)), negative_slope=0.2)
        old = self.classifiers[0].dp1(old)
        old = F.leaky_relu(self.classifiers[0].bn7(self.classifiers[0].linear2(old)), negative_slope=0.2)
        old = self.classifiers[0].dp2(old)
        old = self.classifiers[0].linear3(old)

        # new
        new = F.leaky_relu(self.classifiers[1].bn6(self.classifiers[1].linear1(x)), negative_slope=0.2)
        new = self.classifiers[1].dp1(new)
        new = F.leaky_relu(self.classifiers[1].bn7(self.classifiers[1].linear2(new)), negative_slope=0.2)
        new = self.classifiers[1].dp2(new)
        new = self.classifiers[1].linear3(new)
        return old, new


"""
lwf model for 3D data - DGCNN
- using matrix multiplication with w2v's transformation
"""
class DGCNNLwf3Dv2_2(nn.Module):
    def __init__(self, shared_model, k=20, dropout=0.5, att_size=300):
        super(DGCNNLwf3Dv2_2, self).__init__()
        self.k = k
        for param in shared_model.parameters():
            param.requires_grad = True
        self.bn1 = shared_model.bn1
        self.bn2 = shared_model.bn2
        self.bn3 = shared_model.bn3
        self.bn4 = shared_model.bn4
        self.bn5 = shared_model.bn5
        self.conv1 = shared_model.conv1
        self.conv2 = shared_model.conv2
        self.conv3 = shared_model.conv3
        self.conv4 = shared_model.conv4
        self.conv5 = shared_model.conv5
        linear1 = shared_model.linear1
        bn6 = shared_model.bn6
        dp1 = shared_model.dp1
        linear2 = shared_model.linear2
        bn7 = shared_model.bn7
        dp2 = shared_model.dp2

        self.classifiers = nn.ModuleList([
            nn.ModuleDict({
                'linear1': linear1,
                'bn6': bn6,
                'dp1': dp1,
                'linear2': linear2,
                'bn7': bn7,
                'dp2': dp2,
            }),
            nn.ModuleDict({
                'linear1': nn.Linear(linear1.in_features, 512, bias=False),
                'bn6': nn.BatchNorm1d(512),
                'dp1': nn.Dropout(p=dropout),
                'linear2': nn.Linear(512, 256),
                'bn7': nn.BatchNorm1d(256),
                'dp2': nn.Dropout(p=dropout),
            })
        ])

        self.old_att = shared_model.old_att

        self.new_att = nn.Sequential(
            nn.Linear(att_size, 256),
            # nn.Tanh(),
            nn.ReLU(), 
        )

        self.new_att.apply(init_weights)
        self.classifiers[1].apply(init_weights)

    def forward(self, x, old_att=None, new_att=None):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # old
        old = F.leaky_relu(self.classifiers[0].bn6(self.classifiers[0].linear1(x)), negative_slope=0.2)
        old = self.classifiers[0].dp1(old)
        old = F.leaky_relu(self.classifiers[0].bn7(self.classifiers[0].linear2(old)), negative_slope=0.2)
        old = self.classifiers[0].dp2(old)
        old = F.linear(old, self.old_att(old_att))
        
        # new
        new = F.leaky_relu(self.classifiers[1].bn6(self.classifiers[1].linear1(x)), negative_slope=0.2)
        new = self.classifiers[1].dp1(new)
        new = F.leaky_relu(self.classifiers[1].bn7(self.classifiers[1].linear2(new)), negative_slope=0.2)
        new = self.classifiers[1].dp2(new)
        new = F.linear(new, self.new_att(new_att))

        return old, new