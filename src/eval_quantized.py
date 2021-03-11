'''
 Evaluate the quantized model
 Author: chao.zhang
'''

import os
 
# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from load_data import get_mnist_dataset

MODEL_DIR = './models'
QAUNT_MODEL = 'quantized_model.h5'

# Load the quantized model
print('\nLoad quantized model..')
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
with vitis_quantize.quantize_scope():
    model = models.load_model(path)

# get Mnist dataset
print("\nLoad Mnist dataset..")
(_, _, test_dataset) = get_mnist_dataset()

# Compile the model
print('\nCompile model..')
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

# Evaluate model with test data
print("\nEvaluate model on test Dataset")
loss, acc = model.evaluate(test_dataset)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)
