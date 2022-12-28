from torch.utils.data import Dataset
import pandas
import os
import cv2
import numpy as np
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



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


def get_Dataset(Dataset_name, annotation,root,status="train", N_class=10, transforms=None): 
    if status=='val':
        Valmod=True
    else:
        Valmod=False
    
    
    if Dataset_name=="imagewoof":
        dataset=MyDatasetimagewoof(root,annotation,N_class,Valmod,transforms)
    elif Dataset_name=="cifar10":
        dataset = tv.datasets.CIFAR10(
                    root=root,   # Извлеките загруженный пакет сжатия набора данных в каталог DataSet в текущем каталоге
                    train= not Valmod, 
                    download=False,    # Если вы ранее не загружали набор данных вручную, измените его на True здесь
                    transform=transforms)
    else:
        print("=Датасет не найден")

    return dataset