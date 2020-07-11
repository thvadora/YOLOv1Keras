import numpy as np
import math
import os
from PIL import Image
from tensorflow.keras.utils import Sequence
from preprocessing import TRAIN, VAL
import _pickle as cPickle

class YoloSequenceData(Sequence):

        def __init__(self, target, batch_size):
            if target == 'train':
                self.path = TRAIN
            else:
                self.path = VAL
            self.x = os.listdir(os.path.join(self.path,'X'))
            self.y = os.listdir(os.path.join(self.path,'Y'))
            self.batch_size = batch_size

        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = self.x[idx*self.batch_size : (idx+1)*self.batch_size]
            batch_y = self.y[idx*self.batch_size : (idx+1)*self.batch_size]
            X = np.array([np.array(Image.open(os.path.join(self.path,'X',file_name))) for file_name in batch_x])
            Y = np.array([cPickle.load(open(os.path.join(self.path,'Y',file_name), "rb")) for file_name in batch_y])
            return X,Y
