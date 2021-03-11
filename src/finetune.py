'''
Model finetuning.
Tensorflow 2.3

Author: chao.zhang
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
from load_data import get_mnist_dataset

MODEL_DIR = './models'
FLOAT_MODEL = 'float_model.h5'
FINETUNE_MODEL = 'finetuned_model.h5'

# Load the floating point trained model
print('\nLoad float model..')
path = os.path.join(MODEL_DIR, FLOAT_MODEL)
try:
    model = models.load_model(path)
except:
    print('\nError:load float model failed!')

# get Mnist dataset
print("\nLoad Mnist dataset..")
(train_dataset, val_dataset, test_dataset) = get_mnist_dataset()

# Call Vai_q_tensorflow2 api to create the quantize training model
print("\nCreate quantize training model..")
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
model = quantizer.get_qat_model()

# Compile the model
print("\nCompiling model..")
model.compile(optimizer="rmsprop", 
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

# Train/finetune the model 
print("\nFit on Dataset..")
history = model.fit(train_dataset, epochs=10,validation_data=val_dataset)

# Save finetuned model
path = os.path.join(MODEL_DIR, FINETUNE_MODEL)
model.save(path)
print('\nSaved finetuned model as',path)

# Evaluate model with test data
print("\nEvaluate model on test Dataset..")
loss, acc = model.evaluate(test_dataset)  # returns loss and metrics
print("loss: %.3f" % loss)
print("acc: %.3f" % acc)


