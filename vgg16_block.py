import os

'''这份代码是编写vgg主干网络的代码，vgg网络结合残差神经网络，将卷积层直接与输出曾想加'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt
import pandas
'''
继承Layer基本属性自定义类
'''


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 控制通过捷径的参数和通过普通道路的参数尺寸一样
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

        self.ac2 = layers.Activation('relu')

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.ac1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入中的步长为1，则identity = inputs，否则，需要经过一层卷积网络调整size
        identity = self.downsample(inputs)

        output = self.ac2(identity + out)
        return output


'''
继承Model基本属性创造基本模型
'''


class VGG(keras.Model):
    def __init__(self, layer_dims, num_classes=7):
        super(VGG, self).__init__()

        self.prev = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1))
        ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.layer5 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(512)
        self.droupout=layers.Dropout(0.6)
        self.ac=layers.Activation(activation='relu')
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.prev(inputs)  # initialize data
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = self.fc1(x)
        x=self.droupout(x)
        x=self.ac(x)
        output = self.fc2(x)

        return output  # 到此为止整个残差网络构建完毕

    def build_resblock(selfself, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for i in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def vgg16():
    return VGG([2, 2, 2, 2])



def residual_block(x, output_channel):
    """
    这个函数用来判断输入的通道数和输出的通道数是不是一致的，如果不一致，需要进行降采样，这样才能相加
    """

    input_channel = x.get_shape().as_list()[-1]
    if input_channel == output_channel:
        strides = (1, 1)
        increase_dim = False
    elif input_channel * 2 == output_channel:
        strides = (2, 2)
        increase_dim = True
    else:
        raise Exception("input channel can't match output channel")

    return strides, increase_dim
