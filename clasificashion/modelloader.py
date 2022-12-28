# -*- coding: utf-8 -*-
from torch import device
from torch import load

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
    else :
        print("Модель не найдена")
    
    devices = device("cuda:0" if is_available() else "cpu")        
    model_ft = model_ft.to(devices)
    
    if path != None:
        weights = load(path)
        model_ft.load_state_dict(weights['state_dict'], strict=True)
        model_ft = model_ft.eval()
    
    return model_ft