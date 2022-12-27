from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import transform
import  Dataset as MyDataset

def dataloader(root, annotation, status, Dataset_name,SIZE, N_class,Dataset=None, transforms = None,metod_aug_name=None,rootAA=None, batch_size = 100, num_workers=8, pin_memory=True, drop_last=True, shuffle=False, rasp_file = False ):
    if (transforms != None):
        val_transforms, train_transforms = transforms             
    elif metod_aug_name != "baseline":
        val_transforms, train_transforms =transform.get_transform(metod_aug_name,rootAA,SIZE)
    else:
        val_transforms, train_transforms = transform.get_base_ransforms(SIZE)
       
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