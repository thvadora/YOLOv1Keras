import argparse
from keras.engine import Input
from keras.models import Model
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from sequence import YoloSequenceData

parser = argparse.ArgumentParser(description='Train NetWork.')
parser.add_argument('--epochs', help='Num of epochs.')
parser.add_argument('--batch_size', help='Num of batch size.')

def train(args):
    epochs = int(os.path.expanduser(args.epochs))
    batch_size = int(os.path.expanduser(args.batch_size))

    input_shape = (448, 448, 3)
    inputs = Input(input_shape)
    #out = model(inputs)

    #model = Model(inputs=inputs, outputs=out)
    #model.compile(loss=loss, optimizer=)

    train_generator = YoloSequenceData('train', batch_size)
    validation_generator = YoloSequenceData('val', batch_size)


train(parser.parse_args())
