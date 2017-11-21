import os
import argparse
import numpy as np

from PIL import Image
import tensorflow as tf
import keras

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input

#from tqdm import tqdm
import glob
import matplotlib.pyplot as plt


def get_nb_files(directory):
        """Get number of files by searching directory recursively"""
        cnt = 0
        for r, dirs, files in os.walk(directory):
            for dr in dirs:
                cnt += len(glob.glob(os.path.join(r, dr + "/*")))
        return cnt


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x) 
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

    
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(optimizer='adam',    
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])



def vgg_train(train_dir, val_dir):
  
    train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
    
    test_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE,
        class_mode="binary"
    )
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE,
        class_mode="binary"
    )

    base_model = VGG16(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, nb_classes)

    setup_to_transfer_learn(model, base_model)

    history = model.fit_generator(train_generator, epochs = NB_EPOCHS, steps_per_epoch = nb_train_samples,                                      validation_data=validation_generator, validation_steps = nb_val_samples)
    model.save("vgg_train.model")
    
    return history
    
    
def predict(test_dir):
    from keras.models import load_model
    test_datagen = ImageDataGenerator(vertical_flip=True)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BAT_SIZE,
        class_mode="binary"
    )
    
    model = load_model('vgg_train.model')
    pred = model.predict_generator(self, generator, steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    
    return pred
    

with tf.device('/device:GPU:0'):

    train_dir = 'train_CAL/'
    test_dir = 'test_CAL/'


    IM_HEIGHT = 224
    IM_WIDTH = 224
    NB_EPOCHS = 1
    BAT_SIZE = 16
    FC_SIZE = 500 # May need to train this parameter
    nb_classes = 1


    nb_train_samples = get_nb_files(train_dir)
    nb_classes = len(glob.glob(train_dir + "/*"))
    nb_val_samples = get_nb_files(test_dir)


    h = vgg_train(train_dir, test_dir)


sess = tf.Session(config=tf.ConfigProto())
print(sess.run(h))