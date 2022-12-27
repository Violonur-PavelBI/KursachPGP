# -*- coding: utf-8 -*-
from torch import device
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda import is_available
import timm


def search(list, platform):
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False

def get_model(model_name,N_class=10,path=None,pretrained=False):
    model_list_names = timm.list_models(pretrained=pretrained)
    if search(model_list_names,model_name):
        model_ft = timm.create_model(model_name, pretrained=pretrained,num_classes=N_class)
    elif model_name=="resnet50":
        model_ft= ResNet50(N_class,pretrained)
#     elif model_name=="resnet50":
#         model_ft = models.CNN(N_class,timm_model_name,zagr)
    else :
        print("Модель не найдена")
    
    devices = device("cuda:0" if is_available() else "cpu")        
    model_ft = model_ft.to(devices)
    
    if path != None:
        weights = torch.load(path)
        model_ft.load_state_dict(weights['state_dict'], strict=True)
        model_ft = model_ft.eval()
    
    return model_ft