# -*- coding: utf-8 -*-
import time
import copy
import torch
import numpy as np
from show import show
from tqdm import tqdm
import os
def train_model(model, 
                classification_criterion, 
                optimizer, 
                dataloaders,
                scheduler,
                batch_size, 
                snp_path, 
                Name_experement, 
                num_epochs,
                N_class,
                multiclass=False):
    # Запомнить время начала обучения
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    mass = [[],[],[]]
    t_klass = [0]*N_class
    v_klass = [0]*N_class
    # Копировать параметры поданной модели
    best_model_the_loss_classification = copy.deepcopy(model.state_dict())
    best_Loss_classification = 10000.0         # Лучший покозатель модели
    best_epoch_classification = 0
    best_acc = 0         # Лучший покозатель модели
    best_epoch_classification = 0
    best_epoch_acc=0
    epoch_acc=0
    for epoch in range(num_epochs):
        # У каждой эпохи есть этап обучения и проверки

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Установить модель в режим обучения
            elif phase == 'V':
                model.eval()   #Установить модель в режим оценки
            
            # Обнуление параметров
            running_classification_loss = 0.0
            running_corrects = 0
            dataset_sizes=0
            # Получать порции картинок и иx классов из датасета
            
            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), 
                        desc="Epocha "+phase+" "+str(epoch+1)+"/"+str(num_epochs))
            for step, (inputs, labels) in pbar:
                
                # a = np.zeros((batch_size,N_class))
                # for i in range(batch_size):
                #     for j in range(N_class):
                #         #a[i][int(j)] = 1
                #         if phase == 'T':
                #             t_klass[j]+=classification_label[i][j]
                #         else:
                #             v_klass[j]+=classification_label[i][j]
                # if multiclass == False:
                #     mult_class_labels = np.zeros((batch_size,N_class))
                #     for i in range(batch_size):
                #         for j in range(N_class):
                #             if classification_label[i] == j+1:
                #                 mult_class_labels[i][j]=1
                
                # считать все на видеокарте или ЦП
                inputs = inputs.to(device)
                labels = labels.to(device)
#                 model = model.to(device)
#                 print(labels)
                # обнулить градиенты параметра
                optimizer.zero_grad()

                # forward
                # Пока градиент можно поcчитать, cчитать только на учимся
                with torch.set_grad_enabled(phase == 'train'):
                    # Проход картинок через модель
                    classification = model(inputs)
#                     print(classification)
                    # Получить индексы максимальных элементов
                    
                    # preds = classification.detach().clone()
                    # # print(preds)
                    # for i in range(len(preds)):
                    #     for j in range(len(preds[i])):
                    #         if  preds[i][j] > 0:
                    #             preds[i][j] = 1
                    #         else:
                    #             preds[i][j] = 0
                    # print(preds)
                    # print(torch.argmax(classification,1))
                    
                    total_classification_loss =  classification_criterion(classification, labels.to(dtype=torch.long))
                    loss =  total_classification_loss 
                    # Если учимся
                    if phase == 'train':
                        # Вычислить градиенты
                        total_classification_loss.backward()
                        # Обновить веса
                        optimizer.step()
                
                # Статистика    
                for i in range(batch_size):# Колличество правильных ответов
                    running_corrects += float(torch.sum(torch.argmax(classification[i]) == labels[i]))
                running_classification_loss += total_classification_loss.item() * inputs.size(0)          
                dataset_sizes =dataset_sizes+ batch_size
                epoch_classification_loss = running_classification_loss / dataset_sizes
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(valid_loss=f'{epoch_classification_loss:0.4f}',
                                 acc=f'{epoch_acc:0.5f}',
                                 lr=f'{current_lr:0.5f}',
                                 gpu_memory=f'{mem:0.2f} GB')
                
             # Обновить скорость обучения
            if phase == 'train':
                scheduler.step()
                
            # Усреднить статистику
            running_classification_loss/= dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            
        
            if phase == 'train':
                text_file_t_loss = open(os.path.join(snp_path+Name_experement+"_t_loss.txt"), "a")
                text_file_t_acc = open(os.path.join(snp_path+Name_experement+"_t_acc.txt"), "a")
                text_file_t_loss.write(str(epoch+1)+','+str(running_classification_loss)+'\n')
                text_file_t_acc.write(str(epoch+1)+','+str(epoch_acc)+'\n')
            else:
                text_file_v_loss = open(os.path.join(snp_path+Name_experement+"_v_loss.txt"), "a")
                text_file_v_acc = open(os.path.join(snp_path+Name_experement+"_v_acc.txt"), "a")
                text_file_v_loss.write(str(epoch+1)+','+str(round(running_classification_loss,4))+'\n')
                text_file_v_acc.write(str(epoch+1)+','+str(round(epoch_acc,4))+'\n')
            
            if phase == 'val':
                mass[0].append(epoch)
                mass[1].append(epoch_classification_loss)
                mass[2].append(epoch_acc)
                
            # Копироование весов успешной модели на вэйле              
            if phase == 'val' and epoch_classification_loss < best_Loss_classification:                
                best_Loss_classification = epoch_classification_loss
                best_epoch_classification=epoch + 1
                best_model_the_loss_classification = copy.deepcopy(model.state_dict())
                save_name = snp_path + Name_experement + '_' + str(epoch+1) + '_CrossEntropyLoss-' + str(round(best_Loss_classification,4)) + '_checkpoint.pth.tar'
                torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            }, save_name)
                
            if phase == 'val' and best_acc < epoch_acc:
                print('Best val Loss classification: {:4f},Best val Acc classification:{:4f}'.format(best_Loss_classification,best_acc ))
                best_acc=epoch_acc
                best_epoch_acc=epoch + 1
                best_model_the_acc_classification = copy.deepcopy(model.state_dict())
                save_name = snp_path + Name_experement + '_' + str(epoch+1) + '_ACC top1-' + str(round(best_acc,4)) + '_checkpoint.pth.tar'
                torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            }, save_name)   
                
    # Конечное время и печать времени работы
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss classification: {:.4f} epoch {:.0f}  '.format(best_Loss_classification, best_epoch_classification))
    print('Best val Loss accuracy: {:.4f} epoch {:.0f}'.format(best_acc,best_epoch_acc))
    
    show(mass,0,len(mass[0]))
    overfit_model = model
    model1 = model
    model2 = model
    model1.load_state_dict(best_model_the_loss_classification)
    model2.load_state_dict(best_model_the_acc_classification)
    return model1, model2, overfit_model,best_acc