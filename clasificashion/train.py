# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision as tv
from torchvision import datasets, models, transforms
from datetime import datetime
import time
import copy
import numpy as np
import os
import random
import pandas
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import clasificashion.dataloader as MyDLoad
import clasificashion.modelloader as MyModLoad

import clasificashion.train_alg  as train_alg

show = tv.transforms.ToPILImage() # Тензор может быть преобразован в изображение для легкой визуализации

def train(train_dirr,                                           # Путь до папки с train or directori
          val_dirr,                                             # Путь до папки с описанием val
          model_name,# ="resnet50",                             # Название модели timm or пользовательские из кода
          Dataset_name,# ="cifar10",                            # Название Dataset (при сохранении)
          metod_aug_name,#="baseline",#="FsAA",                 # Название metoda (при сохранении)

          train_annotation=None,                                # Путь до файла с описанием train
          val_annotation=None,                                  # Путь до файла с описанием val
          N_class = 10,                                          # Количество классов в задаче
          num_epochs=100,                                        # Количество эпох обучения
          lr=0.00001,                                           # Коэффициент скорости обучения (Learning rate)
          momentum=0.9,
          batch_size = 64,                                     # Размер бача
          snp_path = '/workspace/proj/clasificashion/snp/',     # Путь к папке, в которую сохранять готовые модели
                                     
          pretrained = False,                                   # True - загрузить предобученную модель
          multicl=True,                                         # мультикласовая ли задача
          num_workers=8,                                        #Cколько подпроцессов использовать для загрузки данных
          pin_memory=True,                                      # Ускорить ли загрузки данных с CPU на GPU False если очень маленький набор данных 
          drop_last_v=True,                                     #Отбрасывать ли последний нецелый батч в val
          drop_last_t=True,                                     #Отбрасывать ли последний нецелый батч в train

          transforms = None,                                    # Аугментации val_transforms, train_transforms на основе Albumentation 
          SIZE = 112,                                           # Размер входа (SIZE*SIZE)
    
          # Характеристики оптимизатора
                                       # коэфицент сохранения момента
          nesterov=False,                                         # Примять ли момент Нестерова
          classification_criterion = nn.CrossEntropyLoss(),     # Loss для классификации
          
          rootAA=None,                                           #Путь к загружаемой аугментации для train
          path = None,#"/snp/YesNo_acc_367-0.9265.tar",         # Путь к модели для загрузки
          Dataset=None,                                         # уже созданный обьект датасет созданый из Класса наледника torch.utils.data.Dataset{def__init__(self, annotation,root,transform=None,):;return image,classification_labels}
          t_shuffle = True,                                     # Перемешивать ли train
          v_shuffle = True,                                     # Перемешивать ли val
          v_rasp_file = False,                                  # Название файла, в котором каждая запись в строке означает с какой вероятностью взять соответствующий пример из анотации
          t_rasp_file = False,                                  # Название файла, в котором каждая запись в строке означает с какой вероятностью взять соответствующий пример из анотации
          ):
    
    Name_experement="_"+model_name+"_"+Dataset_name+"_"+metod_aug_name

    
    def set_seed(seed = 10):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return random_state

    # random_state = set_seed()
    
    dataloaders={'train':MyDLoad.dataloader( train_dirr, train_annotation, 'train',Dataset_name, SIZE, N_class,Dataset, transforms,metod_aug_name,rootAA, batch_size = batch_size,num_workers=num_workers,
                                    pin_memory=pin_memory, drop_last=drop_last_t, shuffle = t_shuffle,rasp_file = t_rasp_file),
                'val':MyDLoad.dataloader( val_dirr, val_annotation, 'val',Dataset_name, SIZE, N_class,Dataset, transforms, metod_aug_name,rootAA,batch_size = batch_size,num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last_v, shuffle = v_shuffle,rasp_file = v_rasp_file)} 
    model_ft=  MyModLoad.get_model(model_name,N_class,path,pretrained)
    
    optimizer_ft =optim.AdamW(model_ft.parameters(),lr=lr,betas=(momentum,0.99))
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=100)
    
    model_loss, model_acc, overfit_model ,best_acc = train_alg.train_model(model_ft,
                                                                classification_criterion,
                                                                optimizer_ft,
                                                                dataloaders,
                                                                exp_lr_scheduler,
                                                                batch_size, 
                                                                snp_path,
                                                                Name_experement,
                                                                num_epochs,
                                                                N_class,
                                                                multicl)
    return  model_acc, best_acc,model_loss, overfit_model

def dataloader(val_dirr,anatation):
    
    landmarks_frame = pandas.read_csv(anatation).query("is_valid == "+str(True))
    index=random.randint(1, len(landmarks_frame))
    img_name = os.path.join(val_dirr,str(landmarks_frame.iloc[index,0]))
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    val_transforms = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])
    image= val_transforms(image=image)["image"]

    landmarks=landmarks_frame.iloc[index,1]
    landmarks = np.array(landmarks)
    lable=landmarks
    if lable =='n02086240':
        label_id= 0
    elif lable == 'n02087394' :
        label_id=1
    elif lable == 'n02088364'  :
        label_id=2
    elif lable == 'n02089973' :
        label_id=3
    elif lable == 'n02093754' :
        label_id=4
    elif lable == 'n02096294' :
        label_id=5
    elif lable == 'n02099601' :
        label_id=6
    elif lable == 'n02105641':
        label_id=7
    elif lable == 'n02111889':
        label_id=8
    elif lable == 'n02115641':
        label_id=9
    else :
        label_id=lable
        print("Нет класса из списка представленых классов")
    label_id=np.array(label_id)
    return image, label_id 