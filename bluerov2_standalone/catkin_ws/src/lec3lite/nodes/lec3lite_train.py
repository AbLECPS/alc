# -*- coding: utf-8 -*-

import numpy as np
import datetime
import random
import numpy as np
import csv
import keras, os
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam, SGD


train = True
# train = False

_shuffle = True
_epochs = 200
_batch_size = 16
softmax_threshold = 0.05
echo_threshold = 0.15

begin_time = datetime.datetime.now()

training_folder = "lec3_training"
sensor_range = 30 # m
# data = np.genfromtxt('../' + training_folder + '/' + filename + '.csv', delimiter=',', skip_header=1, comments="#")
data_path = training_folder
x = None
y = None
(_, dirs, _) = next(iter(os.walk(data_path)))
dirs.sort(reverse=True)
print(dirs)
for training_dir in dirs:
    _x = None
    _y = None
    (_, _, filenames) = next(iter(os.walk(data_path+"/"+training_dir)))
    print("Folder: "+str(training_dir))
    for csv_file in filenames:    
        if csv_file == "bins.csv":
            # if x is not None:
            #     x = np.append(x, np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#"), axis=0)
            # else:
                _x = np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#")
        elif csv_file == "bins_gt.csv":
            # if y is not None:
            #     y = np.append(y, np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#"), axis=0)
            # else:
                _y = np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#")
        else:
            print("ignoring "+(str(csv_file)))    
    if _y is None:
        # HW data - no GT
        # y = np.where(x > 50, x, 0)    
        _y = _x * 0
        # _y[_x>0.35] = 1
        # _y[_y<1] = 0
        _y_idx = np.where(_x > echo_threshold * 255)  
        if len(_y_idx) > 0:
        #     _y[-1] = 1
        # else:
            # _y[_y_idx[0]] = 1
            _y[_y_idx] = 1
        
        # print(y_test[i,gt_ind])
    else:
        # Sim data with GT
        # Ignoring boot up messages
        _x = _x[_x[:,0]>2000000000]
        _y = _y[_y[:,0]>2000000000]

    if x is None and y is None:
        x = _x
        y = _y
    else:
        x = np.append(x, _x, axis=0)
        y = np.append(y, _y, axis=0)       

    print(" x,y sizes after sync ")
    print(" _x size: "+(str(len(_x))))
    print(" _y size: "+(str(len(_y))))
    print("x size: "+(str(len(x))))
    print("y size: "+(str(len(y))))
    print("")

x = x[:,4:]
y = y[:,4:]

print("Total x,y sizes after sync:")
print("x size: "+(str(len(x))))
print("y size: "+(str(len(y))))

# Cleaning out max value, when no obstacle
x[:,-10:] = 0
y[:,-10:] = 0

no_obstacle_data = 0
for i in range(len(y)):    
    if len(y[i,y[i,:]>0]) == 0:
        y[i,-1] = 1        
        no_obstacle_data+=1
print("No obstacle data in x,y: "+str(no_obstacle_data))

# Normalizig
# normalizer_x = 255.0
# normalizer_y = 1152.0


x = np.nan_to_num(x)
y = np.nan_to_num(y)

normalizer_x = 128*2
# normalizer_y = 128*3

# normalizer_x = np.sum(x, axis=1)
normalizer_y = np.sum(y, axis=1)

print("*********\n*********\n*********\n*********\n*********\n*********\n*********\n")
print(normalizer_y[100])

x = x / normalizer_x
# x=np.divide(x, normalizer_x.reshape((normalizer_x.shape[0], -1)))
x[x>1]=1
# y = y / normalizer_y
y=np.divide(y, normalizer_y.reshape((normalizer_y.shape[0], -1)))
y[y>1]=1


x = np.nan_to_num(x)
y = np.nan_to_num(y)






input_shape=(252,)
output_shape = 252
# output_shape = 30
# y = cv2.resize(y, dsize=(output_shape, len(y)), interpolation=cv2.INTER_CUBIC)
# y[y!=0] = 1

# print(x[1000])
# print(y[1000])

print("x size: "+(str(np.shape(x))))
print("y size: "+(str(np.shape(y))))


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.05)

print(x_train.shape)




x_train = x_train.reshape(-1, x.shape[1])
x_test  = x_test.reshape(-1,  x.shape[1])
y_train = y_train.reshape(-1, y.shape[1])
y_test = y_test.reshape(-1, y.shape[1])

# print("Sample X:")
# print(x_train[1000])
print("Sample Y:")
print(y_train[1000])



model = Sequential()
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dense(256, input_shape=input_shape, activation='relu'))
model.add(Dense(64, input_shape=input_shape, activation='relu'))
model.add(Dense(output_shape, input_shape=input_shape, activation='softmax'))

# opt = Adam(learning_rate=0.0001)
# #mean_squared_error
# #categorical_crossentropy
# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
opt = Adam(learning_rate=0.001)
# opt = SGD(learning_rate=0.01, momentum=0.9)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='categorical_crossentropy',optimizer=opt)
# model.compile(loss='mean_squared_error',optimizer=opt)
model.summary()

filename = "lec3lite"
checkpoint_path = '../' + training_folder + '/' + filename + '.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# # =============================================================================
# cp_callback = ModelCheckpoint(filepath=checkpoint_path, 
#                             monitor="val_loss", 
#                             save_best_only=False,
#                             save_weights_only=True, 
#                             verbose=1)


# if train:
#     history = model.fit(x_train, 
#                         y_train, 
#                         epochs=_epochs, 
#                         batch_size=_batch_size, 
#                         callbacks=[cp_callback])    
#     _, train_acc = model.evaluate(x_train, y_train, verbose=1)
#     _, test_acc = model.evaluate(x_test, y_test, verbose=1)

#     print("# =============================================================================")
#     print("train_acc: %f" %train_acc)
#     print("test_acc: %f" %test_acc)

# # =============================================================================
#     model.save(filename+".h5")
# else:
#     model.load_weights(checkpoint_path)
#     # model.summary()


if train:
    print("=============================================================================")
    print("=                               TRAINING                                    =")
    print("=============================================================================")
    checkpoint = ModelCheckpoint(filename, monitor="val_loss",save_best_only=True,save_weights_only=False, verbose=1)
    callbacks=[checkpoint]
    history = model.fit(x_train, 
                        y_train, 
                        epochs=_epochs, 
                        batch_size=_batch_size,
                        shuffle=_shuffle, 
                        callbacks=callbacks)    
    # _, train_acc = model.evaluate(x_train, y_train, verbose=1)
    # _, test_acc = model.evaluate(x_test, y_test, verbose=1)

    # print("# =============================================================================")
    # print("train_acc: %f" %train_acc)
    # print("test_acc: %f" %test_acc)

# =============================================================================
    model.save(filename+".h5")
else:
    print("=============================================================================")
    print("=                        LOADING TRAINED MODEL                              =")
    print("=============================================================================")
    model.load_weights(filename+".h5")
    # model.summary()


print("=============================================================================")
print(datetime.datetime.now() - begin_time)
print("=============================================================================")
# 0/0
random.seed(0)
sensorrange = 30

acc = []

# for i in range(len(x_train)):
#     o = model.predict(x_train[i].reshape(-1, x.shape[1]))
#     acc.append(abs((np.argmax(y_train[i])/output_shape*sensor_range) - (np.argmax(o)/output_shape*sensor_range)))
# print("Train accuracy:")
# print(str(np.mean(acc))+"m, "+str(1-np.mean(acc)/sensor_range))

# for _ in range(100):
    # i = random.randint(0, len(x_test)-1)
for i in range(len(x_test)):
    o = model.predict(x_test[i].reshape(-1, x.shape[1]))
    acc.append(abs((np.argmax(y_test[i])/output_shape*sensor_range) - (np.argmax(o)/output_shape*sensor_range)))
    
    debug_test=False
    debug_test=True
    if debug_test:
        gt_ind = np.where(y_test[i] > echo_threshold)
        print(gt_ind)
        # print(y_test[i,gt_ind])

        o[o<softmax_threshold] = 0
        lec_ind = np.where(o[0,:] > softmax_threshold)
        # for ind in lec_ind:
        #     if ind is in gt_ind:

        print(lec_ind)
        # print(o[0,lec_ind])

        print()

print("Test accuracy:")
print(str(np.mean(acc))+"m, "+str(1-np.mean(acc)/sensor_range))

