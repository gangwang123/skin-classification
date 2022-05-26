# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from config.cfg import train_path

labels = os.listdir(train_path)

# %%
AKIEC, DF, BKL, VASC, NV, BBC, MEL = 0, 0, 0, 0, 0, 0, 0
filename, num = [], []
for file in labels:
    if not file == 'process_data':
        filename.append(file)
        length = len(os.listdir(os.path.join(train_path, file) + '/' + file))
        num.append(length)
    elif file == 'process_data':
        next_filename, next_num = [], []
        for next_file in os.listdir(train_path + 'process_data/'):
            next_length = os.listdir(os.path.join(train_path + 'process_data/', next_file))
            a=0
            for process_file in next_length:
                process_length = os.listdir(os.path.join(os.path.join(train_path + 'process_data/',
                                                                      next_file),process_file))
                length_next=len(process_length)
                a+=length_next
            next_num.append(a)
            next_filename.append(next_file)
# %%
#原始图像比例
plt.figure(figsize=(12,8))
plt.bar(filename,num)
plt.show()
#%%
plt.figure(figsize=(12,8))
plt.bar(filename,[next_num[i]+num[i] for i in range(len(num))])
plt.show()


#%%
#在保证原图的情况下调整比例
#AKIEC的处理
for file in os.listdir(train_path):
    if not file.endswith('process_data'):
        os.mkdir('./dataSet/trainSet_finall/'+file)

#%%
import shutil
for file in os.listdir(train_path):
    if not file=='process_data':
        for f in os.listdir(os.path.join(train_path,file)):
            for ff in os.listdir(os.path.join(os.path.join(train_path,file),f)):
                shutil.copy(os.path.join(os.path.join(os.path.join(train_path,file),f),ff),'./dataSet/trainSet_finall/'+file)

#%%
import shutil
process_path='./dataSet/trainSet/process_data/'
for file in os.listdir(process_path):
    for f in os.listdir(os.path.join(process_path,file)):
        for ff in os.listdir(os.path.join(os.path.join(process_path,file),f)):
            shutil.copy(os.path.join(os.path.join(os.path.join(process_path, file), f), ff),
                        './dataSet/trainSet_finall/' + file)
            print(f"\r {os.path.join(os.path.join(os.path.join(process_path, file), f), ff)}----->{'./dataSet/trainSet_finall/' + file}",end='')

#%%
for file in os.listdir('./dataSet/trainSet_finall'):
    print(file,len(os.listdir(os.path.join('./dataSet/trainSet_finall/',file))))

#%%
#样本均衡处理--NV保持原图
for file in os.listdir('./dataSet/trainSet/NV/NV/'):
    shutil.copy('./dataSet/trainSet/NV/NV/'+file,'./dataSet/trainSet_finall/NV/')


#%%
#DF数据扩大8倍数
#VASC扩大7倍
#BCC扩大两倍

def copy_file(filename,num):
    for i in range(num):
        for file in os.listdir(filename):
            shutil.copy(os.path.join(filename,file),filename+f'{i}'+file)




#%%
copy_file('./dataSet/trainSet_finall/VASC/',3)
#%%
copy_file('./dataSet/trainSet_finall/BCC/',1)


#%%
# for file in os.listdir('./dataSet/trainSet/process_data/DF/'):
#     for f in os.listdir(os.path.join('./dataSet/trainSet/process_data/DF/',file)):
#         shutil.copy(os.path.join(os.path.join('./dataSet/trainSet/process_data/DF/',file),f),'./dataSet/trainSet_finall/DF/')

#%%
copy_file('./dataSet/trainSet_finall/DF/',3)