# Author: Bryce Van Camp
# Project: P10
# File: keras_cnn.py


#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Takes an optional boolean argument and returns the data as described in the 
# specifications.
#
# training - return training data if true, testing data otherwise
def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    
    return (np.reshape(train_images, train_images.shape + (1,)), \
        train_labels) if training else (np.reshape(test_images, \
        test_images.shape + (1,)), test_labels)


# Takes no arguments and returns an untrained neural network as described in 
# the specifications.
def build_model():
    model = keras.Sequential([
        Conv2D(64, 3, activation=Activation('relu'), input_shape=(28,28,1)),
        Conv2D(32, 3, activation=Activation('relu')),
        Flatten(),
        Dense(10, activation=Activation('softmax'))
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


# Takes the model produced by the previous function and the images and labels 
# produced by the first function and trains the data for T epochs; does not 
# return anything.
#
# model - model from build_model
# train_img - training images from get_dataset
# train_lab - training labeels from get_dataset
# test_img - test images from get_dataset
# test_lab - test labels from get_dataset
# T - number of epochs for which to train the data
def train_model(model, train_img, train_lab, test_img, test_lab, T):
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)
    
    model.fit(x=train_img, y=train_lab, validation_data=(test_img, test_lab)\
        , epochs=T)


# Takes the trained model and test images, and prints the top 3 most likely 
# labels for the image at the given index, along with their probabilities.
#
# model - model from build_model with additional Softmax layer
# images - images from get_dataset
# index - specifies which image to consider
def predict_label(model, images, index):
    predictions = model.predict(images)
    
    # get predicted labels for image at given index
    predicted_labels = [(predictions[index][i], class_names[i]) for i in range(len(predictions[index]))]
    
    # sort by highest probability
    predicted_labels.sort(reverse=True, key=lambda tup: tup[0])
    
    for i in range(3):
        print('{}: {:.2f}%'.format(predicted_labels[i][1], predicted_labels[i][0]*100))
    


def main():
    #print(tf.__version__)
    (train_images, train_labels) = get_dataset()
    #print(train_images.shape)
    #print(type(train_images))
    (test_images, test_labels) = get_dataset(False)
    #print(test_images.shape)
    #print(train_images)
    #print(len(train_images))
    #print(len(train_images[0]))
    #print(len(train_images[0][0]))
    #print(train_labels)
    #print(len(train_labels))
    
    model = build_model()
    #keras.utils.plot_model(model, to_file='model.png')
    #print(model)
    #print(model.loss)
    #print(model.optimizer)
    #print(model.metrics_names)
    
    train_model(model, train_images, train_labels, test_images, test_labels, 5)
    
    predict_label(model, test_images, 0)


if __name__ == '__main__':
    main()