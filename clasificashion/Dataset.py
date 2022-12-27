import torch
from torch.utils.data import Dataset
import pandas
import os
import cv2
import numpy as np
import torchvision
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



class MyDataset(Dataset):
    def __init__(self, annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.N_class = N_class

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array(landmarks)
        a = np.zeros(self.N_class)
        for j in landmarks:
            a[int(j)] = 1
        classification_labels = torch.from_numpy(np.array(a).astype('float32'))
#         classification_labels = landmarks.astype('float32').reshape(1)
#         classification_labels = landmarks.astype('float32')
#         if classification_labels>0:
#             classification_labels = np.array(1).astype('float32').reshape(1)
        return image, classification_labels

class HPAIC(Dataset):
    def __init__(self, annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.N_class = N_class

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image_r = cv2.imread(img_name.split('/')[0]+"/"+img_name.split('/')[1]+"/train/"+img_name.split('/')[2]+"_red.png", 0)
        image_g = cv2.imread(img_name.split('/')[0]+"/"+img_name.split('/')[1]+"/train/"+img_name.split('/')[2]+"_green.png", 0)
        image_b = cv2.imread(img_name.split('/')[0]+"/"+img_name.split('/')[1]+"/train/"+img_name.split('/')[2]+"_blue.png", 0)
#         image_y = cv2.imread("train/"+img_name+"_yellow.png")
        image = np.zeros((image_r.shape[0], image_r.shape[1], 3))
        image [:,:,0] = image_r
        image [:,:,1] = image_g
        image [:,:,2] = image_b
        if self.transform:
            image = self.transform(image=image)["image"]
        landmarks = self.landmarks_frame.iloc[idx, 1:]["klass"].split(' ')
        landmarks = np.array(landmarks)
        a = np.zeros(self.N_class)
        for j in landmarks:
            a[int(j)] = 1
        classification_labels = torch.from_numpy(np.array(a).astype('float32'))
        #classification_labels = landmarks.astype('float32')
        #print(image.shape)
#         print(landmarks,classification_labels)
        return image, classification_labels

class retinopatia_2(Dataset):
    def __init__(self, annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.N_class = N_class

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float32')
        if landmarks>0:
            landmarks = np.array(1).astype('float32').reshape(1)
        a = np.zeros(self.N_class)
        for j in landmarks:
            a[int(j)] = 1
        classification_labels = torch.from_numpy(np.array(a).astype('float32'))

        return image, classification_labels

class MyDatasetImagenet(Dataset):
    def __init__(self, annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.N_class = N_class

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root, self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        landmarks = self.landmarks_frame.iloc[idx, 1]
        print(landmarks)
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float32')
        return image, landmarks



class MyDatasetimagewoof(Dataset):
    def __init__(self,root, annotation,n_class, Valid=False, transform=None):    
        self.landmarks_frame = pandas.read_csv(annotation).query("is_valid == "+str(Valid))
        self.transform = transform
        self.valid = Valid  
        self.root=root
        self.N_class=n_class
        
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root,str(self.landmarks_frame.iloc[index,0]))
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image= self.transform(image=image)["image"]

        landmarks=self.landmarks_frame.iloc[index,1]
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
    

class MyDatasetimagenettev2(Dataset):
    def __init__(self,root, annotation,n_class, Valid=False, transform=None):    
        self.landmarks_frame = pandas.read_csv(annotation).query("is_valid == "+str(Valid))
        self.transform = transform
        self.valid = Valid  
        self.root=root
        self.N_class=n_class
        
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root,str(self.landmarks_frame.iloc[index,0]))
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image= self.transform(image=image)["image"]

        landmarks=self.landmarks_frame.iloc[index,1]
        landmarks = np.array(landmarks)
        lable=landmarks
        if lable =='n01440764':
            label_id= 0
        elif lable == 'n02102040' :
            label_id=1
        elif lable == 'n02979186'  :
            label_id=2
        elif lable == 'n03000684' :
            label_id=3
        elif lable == 'n03028079' :
            label_id=4
        elif lable == 'n03394916' :
            label_id=5
        elif lable == 'n03417042' :
            label_id=6
        elif lable == 'n03425413':
            label_id=7
        elif lable == 'n03445777':
            label_id=8
        elif lable == 'n03888257':
            label_id=9
        else :
            label_id=lable
            print("Нет класса из списка представленых классов")
        label_id=np.array(label_id)
        return image, label_id           
    

def get_Dataset(Dataset_name, annotation,root,status="train", N_class=10, transforms=None): 
    if status=='val':
        Valmod=True
    else:
        Valmod=False
    
    if Dataset_name=="cifar10":
        dataset = tv.datasets.CIFAR10(
                    root=root,   # Извлеките загруженный пакет сжатия набора данных в каталог DataSet в текущем каталоге
                    train= not Valmod, 
                    download=False,    # Если вы ранее не загружали набор данных вручную, измените его на True здесь
                    transform=transforms)
    elif Dataset_name=="imagewoof":
        dataset=MyDatasetimagewoof(root,annotation,N_class,Valmod,transforms)
    elif Dataset_name=="Imaginet":
        dataset=MyDatasetImagenet(annotation, root, N_class, transform)
    elif Dataset_name=="imagenettev2":
        dataset=MyDatasetimagenettev2(root,annotation,N_class,Valmod,transforms)
    else:
        print("=Датасет не найден")

    return dataset