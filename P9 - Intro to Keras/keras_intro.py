# Author: Bryce Van Camp
# Project: P9
# File: keras_intro.py


#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Activation, Softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Takes an optional boolean argument and returns the data as described in the 
# specifications.
#
# training - return training data if true, testing data otherwise
def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    return (train_images, train_labels) if training else (test_images, test_labels)


# Takes the dataset and labels produced by the previous function and prints 
# several statistics about the data; does not return anything.
#
# images - images from get_dataset()
# labels - labels from get_dataset()
def print_stats(images, labels):
    class_amts = [0] * 10
    
    print(len(images))
    print('{}x{}'.format(len(images[0]), len(images[0][0])))
    
    # get number of images corresponding to each of the class labels
    for x in labels:
        class_amts[x] += 1
    
    for i in range(len(class_names)):
        print('{}. {} - {}'.format(i, class_names[i], class_amts[i]))


# Takes a single image as an array of pixels and displays an image; does not 
# return anything.
#
# image - single image from get_dataset()
# label - single label from get_dataset()
def view_image(image, label):
    # setup
    fig, ax = plt.subplots()
    ax.set_title(label)
    
    # render img and colorbar
    bar = ax.imshow(image, aspect='equal')
    fig.colorbar(bar, ax=ax)
    
    # render plots
    plt.show()

# Takes no arguments and returns an untrained neural network as described in 
# the specifications.
def build_model():
    model = keras.Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation=Activation('relu')),
        Dense(10)
    ])
    
    model.compile(
        optimizer='adam', 
        loss=SparseCategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy']
    )
    
    return model


# Takes the model produced by the previous function and the images and labels 
# produced by the first function and trains the data for T epochs; does not 
# return anything.
#
# model - model from build_model()
# images - images from get_dataset()
# labels - labels from get_dataset()
# T - number of epochs for which to train the data
def train_model(model, images, labels, T):
    model.fit(x=images, y=labels, epochs=T)


# Takes the trained model produced by the previous function and the test 
# image/labels, and prints the evaluation statistics as described below 
# (displaying the loss metric value if and only if the optional parameter has 
# not been set to False).
#
# model - model from build_model()
# images - images from get_dataset()
# labels - labels from get_dataset()
# show_loss - optional bool for displaying the loss metric value
def evaluate_model(model, images, labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(images, labels)
    
    if show_loss:
        print('Loss: {:.2f}'.format(test_loss))
    
    print('Accuracy: {:.2f}%'.format(test_accuracy*100))


# Takes the trained model and test images, and prints the top 3 most likely 
# labels for the image at the given index, along with their probabilities.
#
# model - model from build_model() with additional Softmax layer
# images - images from get_dataset()
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
    #print(type(train_images))
    #(test_images, test_labels) = get_dataset(False)
    #print(train_images)
    #print(len(train_images))
    #print(len(train_images[0]))
    #print(len(train_images[0][0]))
    #print(train_labels)
    #print(len(train_labels))
    
    #print_stats(train_images, train_labels)
    #print_stats(test_images, test_labels)
    
    view_image(train_images[0], class_names[train_labels[0]])
    #view_image(train_images[9], class_names[train_labels[9]])
    #view_image(train_images[56], class_names[train_labels[56]])
    
    #model = build_model()
    #print(model)
    #print(model.loss)
    #print(model.optimizer)
    #print(model.metrics_names)
    
    #train_model(model, train_images, train_labels, 5)
    
    #evaluate_model(model, test_images, test_labels, show_loss=False)
    #evaluate_model(model, test_images, test_labels)
    
    #model.add(Softmax())
    #predict_label(model, test_images, 0)


if __name__ == '__main__':
    main()