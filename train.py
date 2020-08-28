import argparse
from keras.engine import Input
from keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
from sequence import YoloSequenceData
from model import getModel
from loss import getloss

parser = argparse.ArgumentParser(description='Train NetWork.')
parser.add_argument('--epochs', help='Num of epochs.')
parser.add_argument('--batch_size', help='Num of batch size.')

def train(args):
    epochs = int(os.path.expanduser(args.epochs))
    batch_size = int(os.path.expanduser(args.batch_size))

    #TODO: 
    # Add TensorBoard
    # Code Model
    # code fucking loss f

    checkPointPath = './pretrainedModel/'
    if not os.path.isdir(checkPointPath):
        os.makedirs(checkPointPath)
    modelPath = os.path.join(checkPointPath, 'LastModel.hdf5')
    checkPoint = ModelCheckpoint(
                    modelPath,
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                )

    input_shape = (448, 448, 3)
    inputs = Input(input_shape)

    if os.path.exists(modelPath):
        model = load_model(modelPath)
    else:
        model = getModel(inputs)
        model.compile(loss=getloss, optimizer='adam')

    train_generator = YoloSequenceData('train', batch_size)
    validation_generator = YoloSequenceData('val', batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = len(validation_generator),
        callbacks = [checkPoint],
        verbose = 0
    )


train(parser.parse_args())
