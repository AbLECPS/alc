import keras
from keras.layers import Dense, Flatten, Conv2D, Input


def get_model():
    input = Input(shape=(3,), name='input_data')
    layer1 = Dense(50, activation='relu')(input)
    layer2 = Dense(50, activation='relu')(layer1)
    layer3 = Dense(10, activation='relu')(layer2)
    out = Dense(1, activation='linear')(layer3)
    return keras.models.Model(inputs=input, outputs=out)
