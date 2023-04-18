# import libraries defined in this project
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import tensorflow.keras.metrics
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Conv2D
from tensorflow.keras.optimizers import Adam, SGD



#from callbacks.trainingmonitor import TrainingMonitor
#from callbacks.epochcheckpoint import EpochCheckpoint
import tensorflow.keras.backend as K 
import argparse
import numpy as np
import tensorflow.keras.utils
# from os import path 
# from imutils import paths
import cv2
import tensorflow.keras.backend as K
import tensorflow as tf
import sys
import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Loading input data
print("Loading input data")
training_folder = "lec3_training"
sensor_range = 30 # m
# data = np.genfromtxt('../' + training_folder + '/' + filename + '.csv', delimiter=',', skip_header=1, comments="#")
data_path = training_folder
x = None
y = None
(_, dirs, _) = next(iter(os.walk(data_path)))
print(dirs)
for training_dir in dirs:
    (_, _, filenames) = next(iter(os.walk(data_path+"/"+training_dir)))
    print("Folder: "+str(training_dir))
    for csv_file in filenames:    
        if csv_file == "bins.csv":
            if x is not None:
                x = np.append(x, np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#"), axis=0)
            else:
                x = np.genfromtxt(data_path + "/" + training_dir + "/" + csv_file, delimiter=',', skip_header=1, comments="#")
        else:
            print("ignoring "+(str(csv_file)))    

    # Ignoring boot up messages
    x = x[x[:,0]>2000000000]

    
    

x = x[:,4:]

print("Total x sizes after sync:")
print("x size: "+(str(len(x))))

# Normalizig
normalizer_x = 255.0

x = x / normalizer_x

x = np.nan_to_num(x)

x[:,-15:] = 0

import cv2
input_shape=(252,)
output_shape = 252
# output_shape = 30
# y = cv2.resize(y, dsize=(output_shape, len(y)), interpolation=cv2.INTER_CUBIC)
x[x!=0] = 1

# print(x[1000])
# print(y[1000])

print("x size: "+(str(np.shape(x))))

#############


fname="lec3lite.h5"
model = Sequential()
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dense(512, input_shape=input_shape, activation='relu'))
model.add(Dense(output_shape, input_shape=input_shape, activation='softmax'))
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error',optimizer=opt)

model.load_weights(fname)
# model.save("lec2.h5")
print("Previous input shape: ", model.input.shape)
model.input.set_shape((1,) + model.input.shape[1:])
print("New input shape: ", model.input.shape)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = X
def representative_data_gen():
    # Using training data:
    for xi in x:
      sample_tensor = tf.reshape(tf.convert_to_tensor(xi, dtype=tf.float32), model.input.shape)
    #   print(sample_tensor)
      yield [sample_tensor]
    # for i in range(100):

    # Random values
    #   sample_tensor = tf.random.uniform(model.input.shape)      
    #   sample_tensor = sample_tensor/255
    #   print(sample_tensor)
    #   yield [sample_tensor]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

# Set the optimization mode 
# # # converter.optimizations = [tf.lite.Optimize.DEFAULT]


# # # # Convert and Save the model
# # # tflite_model = converter.convert()
# # # open("lec2.tflite", "wb").write(tflite_model)


with open('lec3_quant.tflite', 'wb') as f:
  f.write(tflite_model)

# in shell:
# edgetpu_compiler lec2_quant.tflite