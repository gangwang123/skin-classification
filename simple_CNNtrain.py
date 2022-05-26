#%%
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import warnings
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
warnings.filterwarnings('ignore')

#%%
def set_GPU():
    """GPU相关设置"""

    # 打印变量在那个设备上
    # tf.debugging.set_log_device_placement(True)
    # 获取物理GPU个数
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('物理GPU个数为：', len(gpus))
    # 设置内存自增长
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # print('-------------已设置完GPU内存自增长--------------')
    # 设置哪个GPU对设备可见，即指定用哪个GPU
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    # 切分逻辑GPU
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],  # 指定要切割的物理GPU
    #     # 切割的个数，和每块逻辑GPU的大小
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096*5),]
    #      # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096), ]
    # )#最后一块物理GPU切分成两块，现在逻辑GPU个数为2
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))
set_GPU()



#%%
base_dir='./dataSet/'
train_path=os.path.join(base_dir,'trainSet')
test_path=os.path.join(base_dir,'validationSet')
forest_path=os.path.join(base_dir,'testSet')


# a=Image.open('./dataSet/trainSet/AKIEC/ISIC_0024329.jpg')
# np.array(a).shape
model=tf.keras.Sequential()

##layer 1
# model.add(tf.keras.layers.Input())
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(224,224, 3)))
# model.add(tf.keras.layers.BatchNormalization())##把激活的数据做归一化，在激活函数前面
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D(2,2,padding='SAME'))


##layer2
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2,padding='SAME'))

##faltten,多维向低维的转换
model.add(tf.keras.layers.Flatten())

##全连阶层
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(7,activation='softmax'))


model.compile(optimizer='adam',metrics=['acc'],loss=['categorical_crossentropy'])

print(model.summary())


# ###数据的预处理--读进来的数据会被自动转换为tensor格式，分别准备训练和测试
# #图像的归一化（0-1）之间
train_dataGen=ImageDataGenerator(rescale=1./255)
test_dataGen=ImageDataGenerator(rescale=1./255)
forest_dataGen=ImageDataGenerator(rescale=1./255)
#
# ##从文件夹中读取数据
# train_generator=train_dataGen.flow_from_directory(train_path,#训练数据路径
#                                                   target_size=(412,412),#制定resize大小,和输入大小想匹配
#                                                   batch_size=32,##一次拿出32张图片来训练
#                                                   class_mode='categorical'##和标签相关，one-hot的话就用categorical，二分类的话可以用binary
#                                                   )
#
# test_generator=train_dataGen.flow_from_directory(test_path,#训练数据路径
#                                                   target_size=(412,412),#制定resize大小
#                                                   batch_size=32,
#                                                   class_mode='categorical'##one-hot的话就用categorical，二分类的话可以用binary
#                                                   )
#
# ##训练网络模型
# #直接fit也是可以的，但是通常不能把所有数据全部放入内存,因为有时候数据特别特别大，fit_generator相当于一个生成器，动态产生所需的batch数据，训练的时候用谁，拿谁，不用全部都读到内存中去
# #step_per_epoch 相当于给定一个停止条件，因为生成器会不断的产生batch数据，也就是说他不知道一个epoch里需要执行多少个step
#
# history=model.fit_generator(train_generator,
#                             steps_per_epoch=len(train_generator),#steps_per_epoch*20=2000
#                             epochs=50,#每个batch训练迭代多少次
#                             validation_data=test_generator,##需不需要做验证
#                             verbose=2,
#                             validation_steps=len(test_generator),#validation_steps*20=10000,
#                             )
#
# train_acc=history.history['acc']
# vali_acc=history.history['val_acc']
#
# train_loss=history.history['loss']
# vali_loss=history.history['val_loss']
#
# plt.figure(figsize=(8,6))
# plt.plot(range(len(train_acc)),train_acc,marker='v',label='train_acc')
# plt.plot(range(len(train_acc)),vali_acc,marker='o',label='vali_acc')
# plt.legend()
# plt.savefig('./output/simple_train_acc')
# plt.show()
#
# plt.figure(figsize=(8,6))
# plt.plot(range(len(train_loss)),train_loss,marker='v',label='train_loss')
# plt.plot(range(len(train_loss)),vali_loss,marker='o',label='vali_loss')
# plt.legend()
# plt.savefig('./output/simple_train_loss')
# plt.show()


#数据增强，在基础训练集的基础上 对 训练集进行二次数据增强操作
###在数据增强之前，很明显存在数据过拟合问题，看一下在数据增强之后，能不能解决数据过拟合问题
###数据的预处理--读进来的数据会被自动转换为tensor格式，分别准备训练和测试
#图像的归一化（0-1）之间
#%%
train_dataGen_deal=ImageDataGenerator(rescale=1./255,rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

test_dataGen_deal=ImageDataGenerator(rescale=1./255)

forest_dataGen_deal=ImageDataGenerator(rescale=1./255)

train_after_generator=train_dataGen_deal.flow_from_directory(train_path,
                                                       target_size=(224,224),
                                                       class_mode='categorical',
                                                       batch_size=16)
vali_data=test_dataGen_deal.flow_from_directory(test_path,
                                          target_size=(224,224),
                                          class_mode='categorical',
                                          batch_size=16)

after_generator=model.fit_generator(train_after_generator,
                                   epochs=30,
                                   steps_per_epoch=len(train_after_generator),
                                   validation_data=vali_data,
                                   verbose=2,
                                    validation_steps=len(vali_data)
                                   )
history=after_generator
train_acc=history.history['acc']
vali_acc=history.history['val_acc']

train_loss=history.history['loss']
vali_loss=history.history['val_loss']

plt.figure(figsize=(8,6))
plt.plot(range(len(train_acc)),train_acc,marker='v',label='train_acc')
plt.plot(range(len(train_acc)),vali_acc,marker='o',label='vali_acc')
plt.legend()
plt.savefig('./output/d_simple_train_acc')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(range(len(train_loss)),train_loss,marker='v',label='train_loss')
plt.plot(range(len(train_loss)),vali_loss,marker='o',label='vali_loss')
plt.legend()
plt.savefig('./output/d_simple_train_loss')

plt.show()

