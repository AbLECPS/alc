This is the LEC3 TFLite codebase. This is a signal processing NN. Input is the 252 bin sensor raw data (digitalized analog signal from sonar, with multiple contact echos)

The lec3_lite_node.py will return with an array of contacts in [m]

tflite inference takes around 2ms in average on x64 CPU (i7-9900)

- this code needs alc_utils either in the path, or in this folder

- lec3_training folder: here are the training data, each set in each folder