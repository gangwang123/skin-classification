# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/home/huangby/detection/黑色素瘤分类', '/home/huangby/detection/黑色素瘤分类'])
from vgg16_block import vgg16
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from config.cfg import train_path, test_path, val_path, labels,all_train_path
from config.cfg import WIDTH,HEIGHT,batch_size


# %%
model = vgg16()

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
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # 切分逻辑GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],  # 指定要切割的物理GPU
        # 切割的个数，和每块逻辑GPU的大小
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096*5),]
         # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096), ]
    )#最后一块物理GPU切分成两块，现在逻辑GPU个数为2
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))

set_GPU()

log_dir = './finall_logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model.build(input_shape=(None, WIDTH, HEIGHT, 3))
print(model.summary())


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'best_model.h5',
                                                monitor='val_loss', save_weights_only=True, save_best_only=True,
                                                period=1)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
learning_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
# print(model.summary())


##把所有图片打乱随机取
# %%
def read_all_pic(filename):
    x_array, y_label = [], []
    for file in os.listdir(filename):
        for img in os.listdir(os.path.join(filename, file)):
            image = Image.open(os.path.join(os.path.join(filename, file), img))
            image=image.resize((WIDTH, HEIGHT))
            img_array = np.array(image)/ 255
            x_array.append(img_array)
            yyyy = np.zeros((1, len(labels)))[0]
            yyyy[labels.index(file)] = 1
            y_label.append(yyyy)
    return x_array, y_label

print('processing trainSet')
trainLLst, y_lst = read_all_pic(all_train_path)
trainLst = list(zip(trainLLst, y_lst))
random.shuffle(trainLst)
# %%

print('processing validationSet')
valiLst, vali_y = read_all_pic(val_path)
valiLst = list(zip(valiLst, vali_y))
random.shuffle(valiLst)

all_pic = trainLst + valiLst
random.shuffle(all_pic)

x_train, y_train = zip(*all_pic)
print(len(x_train), len(y_train))


# for p in all_pic:
#     a,b=zip(*p)
#     x_train.append(a)
#     y_train.append(b)

# %%

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            # -------------------------------------#
            #   读取输入图片并进行归一化和resize
            # -------------------------------------#
            X_train.append(tf.cast(lines[i][0],tf.float32))
            Y_train.append(tf.cast(list(lines[i][1]), tf.int32))
            i = (i + 1) % n
        yield (np.asarray(X_train), np.asarray(Y_train))

#%%
batch_size=4
num_val = int(len(all_pic) * 0.1)
num_train = len(all_pic) - num_val

print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

generator = model.fit_generator(generate_arrays_from_file(all_pic[:num_train], batch_size),
                                steps_per_epoch=max(1, num_train // batch_size),
                                validation_data=generate_arrays_from_file(all_pic[num_train:], batch_size),
                                validation_steps=max(1, num_val // batch_size),
                                epochs=70,
                                initial_epoch=0,
                                callbacks=[checkpoint, learning_reduce, early_stopping])
# generator=model.fit(x_train,y_train,batch_size=batch_size,validation_split=0.1,epochs=50,class_weight='auto')

#%%
history = generator
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(range(1,len(train_loss)+1), train_acc, marker='v', label='train_acc')
plt.plot(range(1,len(train_loss)+1), val_acc, marker='o', label='vali_acc')
plt.legend()
plt.savefig('./output/VGG_train_acc')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1,len(train_loss)+1), train_loss, marker='v', label='train_loss')
plt.plot(range(1,len(train_loss)+1), val_loss, marker='o', label='vali_loss')
plt.legend()
plt.savefig('./output/VGG_train_loss')

plt.show()

#%%
f=open(log_dir+'withoutDeal_logs.txt','a')
f.write('accuracy ')
f.write('val_accuracy ')
f.write('train_loss ')
f.write('val_loss+\n')
result_logs=list(history.history.values())
for l in range(len(result_logs[0])):
    f.write(str(result_logs[0][l])+' ')
    f.write(str(result_logs[1][l])+' ')
    f.write(str(result_logs[2][l])+' ')
    f.write(str(result_logs[3][l])+'\n')


#%%
x_test,y_test=read_all_pic('./dataSet/testSet')

#%%

#%%

