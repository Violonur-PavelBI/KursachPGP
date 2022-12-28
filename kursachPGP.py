# cd C:\Users\pacha\Documents\GitHub\KursachPGP
# mpiexec -np 4 python kursachPGP.py
from mpi4py import MPI
from clasificashion.train import train,dataloader
import torch

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    p = comm.Get_size()

    if my_rank ==0:
        dirr = "/data"
        file = "/data/noisy_imagewoof.csv"
        # 
        # предпроцесинг с путями дб
        # 
        for procid in range(1,p):
            comm.send(dirr,file,dest=procid)
    else:
        dirr,file=comm.recv(source=0)

    SIZE = 112
    train_dirr = val_dirr = dirr
    N_class = 10
    batch_size = 50
    num_epochs = 30
    lr = 0.0001
    if my_rank !=0:
        model_acc ,best_acc, _ = train(train_dirr,                                         # Путь до папки с train or directori
                                            val_dirr,                                             # Путь до папки с описанием val
                                            model_name ="resnet18",                             # Название модели timm or пользовательские из кода
                                            Dataset_name ="imagewoof",                            # Название Dataset (при сохранении)
                                            metod_aug_name="baseline",                # Название metoda (при сохранении)
                                            train_annotation=file,                                # Путь до файла с описанием train
                                            val_annotation=file,                                  # Путь до файла с описанием val
                                            N_class=N_class,                                          # Количество классов в задаче
                                            num_epochs=num_epochs,                                        # Количество эпох обучения
                                            batch_size=batch_size)
        comm.send(best_acc,dest=0)
    else:
        for procid in range(1,p):
            message = comm.recv(source=procid)

    if my_rank ==0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img,lable=dataloader(dirr,file)
        print("True_class:"+str(lable))
        img=img.to(device)
        for procid in range(1,p):
            comm.send(img,dest=procid)
        out_label=comm.recv(source=1)
        for procid in range(2,p):
            out_label+=comm.recv(dest=procid)
        print("Ansmbel_class:"+torch.max(out_label, 1))
    else:
        img = comm.recv(source=0)
        out_label=model_acc(img)
        comm.send(out_label,dest=0)