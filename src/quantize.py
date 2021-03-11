'''
 Quantize the float point model
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
FLOAT_MODEL = 'float_model.h5'
QAUNT_MODEL = 'quantized_model.h5'

# Load the floating point trained model
print('Load float model..')
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
try:
    float_model = models.load_model(path)
except:
    print('\nError:load float model failed!')

# get input dimensions of the floating-point model
height = float_model.input_shape[1]
width = float_model.input_shape[2]

# get Mnist dataset
print("\nLoad Mnist dataset..")
(_, _, test_dataset) = get_mnist_dataset()

# Run quantization
print('\nRun quantization..')
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=test_dataset)

# Save quantized model
path = os.path.join(MODEL_DIR, QAUNT_MODEL)
quantized_model.save(path)
print('\nSaved quantized model as',path)




