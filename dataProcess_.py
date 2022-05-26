#%%
import pandas as pd
import tensorflow as tf
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.cfg import train_path

"""
这份代码是用来做特征工程的，包括图片数据的缩放、旋转、裁剪、反转、平移等等操作
"""


'''
for file in os.listdir(train_path):
    print(file,len(os.listdir(os.path.join(train_path,file))))
'''
#%%
for file in os.listdir(train_path):
    from_path=train_path

    # for f in os.listdir(os.path.join(train_path,file)):

    ##将图片转换为一定的size的图片后保存
    to_path = os.path.join(train_path, file)+'/resize'
    try:
        os.mkdir(to_path)
    except:
        pass
    gen = ImageDataGenerator()
    gen_data = gen.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                       target_size=(224,224),
                                       save_prefix='resize')
    for i in range(len(gen_data)):
        gen_data.next()
    #角度旋转
    to_path = os.path.join(train_path, file) + '/rotation'
    try:
        os.mkdir(to_path)
    except:
        pass
    fit_model = ImageDataGenerator(rotation_range=60)
    gen_data = fit_model.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                             target_size=(224,224),
                                             save_prefix='rotation')
    for i in range(len(gen_data)):
        gen_data.next()

    #平移变换
    to_path = os.path.join(train_path, file) + '/shfit'
    try:
        os.mkdir(to_path)
    except:
        pass
    fit_model = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3)  ##W和H参数是0-1之间的数值，相对长度和宽度的比例值
    gen_data = fit_model.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                             target_size=(224,224),
                                             save_prefix='shfit')
    for i in range(len(gen_data)):
        gen_data.next()

    fit_model = ImageDataGenerator(zoom_range=0.3)  # 缩放比例,0-1放大，大于1--缩小
    to_path = os.path.join(train_path, file) + '/zoom'
    try:
        os.mkdir(to_path)
    except:
        pass
    gen_data = fit_model.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                             target_size=(224,224),
                                             save_prefix='zoom')
    for i in range(len(gen_data)):
        gen_data.next()
    #√channel-shfit--对颜色通道进行变换
    to_path = os.path.join(train_path, file) + '/channel'
    try:
        os.mkdir(to_path)
    except:
        pass
    fit_model = ImageDataGenerator(channel_shift_range=15)  #
    gen_data = fit_model.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                             target_size=(224,224),
                                             save_prefix='channel')
    for i in range(len(gen_data)):
        gen_data.next()

    fit_model = ImageDataGenerator(horizontal_flip=True)  ##水平翻转/垂直翻转
    to_path = os.path.join(train_path, file) + '/flip'
    try:
        os.mkdir(to_path)
    except:
        pass
    gen_data = fit_model.flow_from_directory(from_path, batch_size=1, shuffle=False, save_to_dir=to_path,
                                             target_size=(224,224),
                                             save_prefix='flip')
    for i in range(len(gen_data)):
        gen_data.next()


