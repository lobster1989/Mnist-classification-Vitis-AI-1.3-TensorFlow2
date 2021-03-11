'''
Create & train a custom cnn model for Mnist classication
Tensorflow 2.3

Author: chao.zhang
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from load_data import get_mnist_dataset

MODEL_DIR = './models'
FLOAT_MODEL = 'float_model.h5'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

def customcnn():
    # create a cnn model
    inputs = keras.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_customcnn_model')
    model.summary()

    # Compile the model
    model.compile(optimizer="rmsprop", 
            loss="categorical_crossentropy",
            metrics=['accuracy']
            )

    return model

# get Mnist dataset
print("\nLoad Mnist dataset..")
(train_dataset, val_dataset, test_dataset) = get_mnist_dataset()

# build cnn model
print("\nCreate custom cnn..")
model = customcnn()

# Train the model for 10 epochs using a dataset
print("\nFit on dataset..")
history = model.fit(train_dataset, epochs=10,validation_data=val_dataset)

# Save model
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
print("\nSave trained model to{}.".format(path))
model.save(path)

# Evaluate model with test data
print("\nEvaluate model on test dataset..")
loss, acc = model.evaluate(test_dataset)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)


