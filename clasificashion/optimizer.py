# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def get_optimizer(model,lr,momentum,nesterov,step_size,gamma):
    
    # Критерий ошибки CrossEntropy
    # Обратите внимание, что все параметры оптимизируются
#     optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum,nesterov=nesterov)
#     optimizer_ft = optim.RAdam(model.parameters(), lr=lr,(momentum,0.99),eps=eps)
#     optimizer_ft =optim.AdamW(model.parameters(), lr=lr,betas=(momentum,0.99))
    optimizer_ft =optim.AdamW(model.parameters(),lr=lr,betas=(momentum,0.99))

    # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=100)
    return optimizer_ft, exp_lr_scheduler