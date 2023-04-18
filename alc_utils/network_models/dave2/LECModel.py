import keras
from keras.layers import Dense, Flatten, Conv2D, Input


def get_model():
    # model = keras.models.Sequential()
    input1 = Input(shape=(66, 200, 3), name='image')
    layer1 = Conv2D(24, (5, 5), padding="valid",
                    strides=(2, 2), activation="relu")(input1)
    layer2 = Conv2D(36, (5, 5), padding="valid",
                    strides=(2, 2), activation="relu")(layer1)
    layer3 = Conv2D(48, (5, 5), padding="valid",
                    strides=(2, 2), activation="relu")(layer2)
    layer4 = Conv2D(64, (3, 3), padding="valid",
                    strides=(1, 1), activation="relu")(layer3)
    layer5 = Conv2D(64, (3, 3), padding="valid", strides=(
        1, 1), activation="relu", name="test")(layer4)
    layer6 = Flatten()(layer5)
    layer7 = Dense(1164, activation='relu')(layer6)
    layer8 = Dense(100, activation='relu')(layer7)
    layer9 = Dense(50, activation='relu')(layer8)
    layer10 = Dense(50, activation='relu')(layer9)
    layer11 = Dense(10, activation='relu')(layer10)
    heading_out = Dense(1, activation='linear')(layer11)
    return keras.models.Model(inputs=input1, outputs=heading_out)


class CustomFunction:
    def __init__(self, model=None):
        self.model = model
        pass

    def custom_function(self, true, pred):
        test_layer = self.model.get_layer("test")
