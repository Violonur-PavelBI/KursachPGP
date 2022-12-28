from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import  clasificashion.Dataset as MyDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def dataloader(root, annotation, status, Dataset_name,SIZE, N_class,Dataset=None, transforms = None,metod_aug_name=None,rootAA=None, batch_size = 100, num_workers=8, pin_memory=True, drop_last=True, shuffle=False, rasp_file = False ):
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
    if status == "train":
        transforms = train_transforms
    elif status == "val":
        transforms = val_transforms
    
    sampler = None
    if not not rasp_file:
        rasp_file = open(dirr + "/" + rasp_file, "r")
        rasp_ist = rasp_file.readlines()
        rasp_ist = [float(rasp_ist[i]) for i in range(len(rasp_ist))]
        sampler = WeightedRandomSampler(rasp_ist, len(rasp_ist))
    
    if Dataset != None:
        dataset=Dataset
    else:
        dataset= MyDataset.get_Dataset(Dataset_name, annotation,root,status, N_class, transforms)
    
    dataloaders=DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,sampler=sampler,
                                                num_workers=num_workers,drop_last=drop_last,pin_memory=pin_memory)
    return dataloaders