from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras import backend as K
from keras.models import Model

def getModel(inputs):
    out = Flatten()(inputs)
    out = Dense(7*7*30, activation='relu')(out)
    out = Reshape((7, 7, 30))(out)
    return Model(inputs=inputs, outputs=out)