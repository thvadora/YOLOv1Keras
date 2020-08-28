import tensorflow as tf
from tensorflow.keras import backend as K

def getloss(ytrue, ypred):
    return K.sum(ypred)+K.sum(ytrue)