import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Distillation Loss (Hinton)
"""
class SoftTargetKDLoss(nn.Module):
    def __init__(self, T=2.0, scale=1.0):
        super(SoftTargetKDLoss, self).__init__()
        self.T = T
        self.scale = scale

    def forward(self, out_student, out_teacher):
        loss = F.kl_div(F.log_softmax(out_student/self.T, dim=1), F.softmax(out_teacher/self.T, dim=1), reduction='batchmean') * self.T * self.T
        return loss

