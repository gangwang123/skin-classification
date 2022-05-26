import os
train_path = "F:\dataSet/trainSet/"
test_path = "F:\dataSet/testSet/"
val_path = 'F:\dataSet/validationSet/'
all_train_path='F:\dataSet/trainSet_finall/'
labels=os.listdir(all_train_path)

WIDTH= 224
HEIGHT = 224
N_CLASSES = 7
batch_size = 4
