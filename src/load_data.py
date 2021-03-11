'''
Load Mnist dataset from Keras 
Tensorflow 2.3

Author: chao.zhang
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
# import numpy as np

batch_size = 64

def get_mnist_dataset():
    # Get the data as Numpy arrays
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Data preprocessing
    x_train = x_train.reshape((60000,28,28,1)).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train)

    x_test = x_test.reshape((10000,28,28,1)).astype('float32') / 255
    y_test = keras.utils.to_categorical(y_test)

    # Create saperated datasets for train,validate,test
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:50000], y_train[:50000])).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_train[50000:], y_train[50000:])).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
    return (train_dataset, val_dataset, test_dataset)

