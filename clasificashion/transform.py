# -*- coding: utf-8 -*-
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_ransforms(SIZE):
    val_transforms = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])
    
    train_transforms =A.Compose([
        A.Resize(int(SIZE*1.05), int(SIZE*1.05)),
        A.CenterCrop(SIZE,SIZE),
#         A.augmentations.transforms.ColorJitter(),
        A.HorizontalFlip(p=0.5),#Отразите вход по горизонтали, вертикали или по горизонтали и вертикали.
#         A.Rotate(p=0.5),#Поверните ввод на угол, случайно выбранный из равномерного распределения. 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])    
    return val_transforms, train_transforms

def get_transform(metod_aug_name,rootAA,SIZE):
    if metod_aug_name == "FsAA":
        val_transforms = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
        ])
        
        train_transforms =A.load(rootAA)   
    elif metod_aug_name == "PBA":
        val_transforms, train_transforms =get_base_ransforms(SIZE)
    else:
        print("Аугментации не найдена")
    return val_transforms, train_transforms

