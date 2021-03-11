# Mnist-classification-Vitis-AI-1.3-TensorFlow2
Vitis-AI 1.3 TensorFlow2 flow with a custom CNN model, targeted ZCU102 evaluation board.

## Introduction
The whole project should be run in Vitis-AI 1.3 Docker, vitis-ai-tensorflow2 conda environment.Follow https://github.com/Xilinx/Vitis-AI to setup the environment before starting.

This tutorial is quite similar to the Xilinx released tutorial https://github.com/Xilinx/Vitis-AI-Tutorials/tree/MNIST-Classification-TensorFlow, except this tutorial is using VAI 1.3 and Tensorflow 2.3, while the Xilinx released turotial is using VAI 1.2 and Tensorflow 1.15. Also finetuning is introduced in this turorial.

Refer to http://yann.lecun.com/exdb/mnist/ for the Mnist hand-written digits dataset.

We will run the following steps:

* Training and evaluation of a small custom convolutional neural network using TensorFlow2.
* Quantization of the floating-point model.
* Evaluation of the quantized model.
* Apply finetuning to the trained model with a calibration dataset.
* Compilation of both the quantized & finetuned model to create the .xmodel files ready for execution on the DPU accelerator IP.
* Download and run the application on the ZCU102 evaluation board.

## Python scripts
*load_data.py*: 
load Mnist dataset

*generate_images.py*: 
Generate local images from Keras datasets. This file is form https://github.com/Xilinx/Vitis-AI-Tutorials/tree/MNIST-Classification-TensorFlow 

*train.py*: 
Create & train a simple CNN model for Mnist classification. A trained floating point model will be saved.

*quantize.py*: 
Quantize the saved floating point model with Vitis Quantizer. A quantized model will be saved.

*eval_quantized.py*: 
Evaluate the quantized model.

*finetune.py*: 
Model finetuning.

## Shell scripts
*compile.sh*: 
Launches the vai_c_tensorflow2 command to compile the quantized or finetuned model into an .xmodel file for the ZCU102 evaluation board

*make_target.sh*: 
Copies the .xmodel and images to the ./target_zcu102 folder ready to be copied to the ZCU102 evaluation board's SD card.
