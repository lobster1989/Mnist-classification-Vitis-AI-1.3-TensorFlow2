# Mnist-classification-Vitis-AI-1.3-TensorFlow2
Vitis-AI 1.3 TensorFlow2 flow with a custom CNN model, targeted ZCU102 evaluation board.

## Introduction
The project should be run in Vitis-AI 1.3 Docker, vitis-ai-tensorflow2 conda environment.Follow https://github.com/Xilinx/Vitis-AI to setup the environment before starting.

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

## Implement
Before running this part, we should setup Vitis-AI docker and activate vitis-ai-tensorflow2 anaconda environment.
For more details, refer to the latest version of the Vitis AI User Guide (UG1414). 

### Build and train model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python train.py 

Load Mnist dataset..

Create custom cnn..
Model: "mnist_customcnn_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 576)               0         
_________________________________________________________________
dense (Dense)                (None, 64)                36928     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________

Fit on dataset..
Epoch 1/10
782/782 [==============================] - 13s 16ms/step - loss: 0.1843 - accuracy: 0.9427 - val_loss: 0.0701 - val_accuracy: 0.9813
Epoch 2/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0529 - accuracy: 0.9835 - val_loss: 0.0543 - val_accuracy: 0.9855
Epoch 3/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0346 - accuracy: 0.9894 - val_loss: 0.0472 - val_accuracy: 0.9877
Epoch 4/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0252 - accuracy: 0.9929 - val_loss: 0.0463 - val_accuracy: 0.9878
Epoch 5/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0188 - accuracy: 0.9945 - val_loss: 0.0494 - val_accuracy: 0.9896
Epoch 6/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0147 - accuracy: 0.9956 - val_loss: 0.0513 - val_accuracy: 0.9901
Epoch 7/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0121 - accuracy: 0.9966 - val_loss: 0.0452 - val_accuracy: 0.9920
Epoch 8/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0096 - accuracy: 0.9973 - val_loss: 0.0542 - val_accuracy: 0.9906
Epoch 9/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0088 - accuracy: 0.9976 - val_loss: 0.0640 - val_accuracy: 0.9893
Epoch 10/10
782/782 [==============================] - 5s 7ms/step - loss: 0.0073 - accuracy: 0.9978 - val_loss: 0.0709 - val_accuracy: 0.9893

Save trained model to./models/float_model.h5.

Evaluate model on test dataset..
157/157 [==============================] - 1s 3ms/step - loss: 0.0426 - accuracy: 0.9911
loss: 0.043
acc: 0.991
```

### Quantize the floating-point model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python quantize.py 
Load float model..

Load Mnist dataset..

Run quantization..
[INFO] Start CrossLayerEqualization...
10/10 [==============================] - 0s 32ms/step
[INFO] CrossLayerEqualization Done.
[INFO] Start Quantize Calibration...
157/157 [==============================] - 12s 77ms/step
[INFO] Quantize Calibration Done.
[INFO] Start Generating Quantized Model...
[Warning] Skip quantize pos adjustment for layer quant_dense, its quantize pos is [i=None, w=8.0, b=9.0, o=3.0]
[INFO] Generating Quantized Model Done.

Saved quantized model as ./models/quantized_model.h5
```

### Evaluate quantized model

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python eval_quantized.py 

Load quantized model..
WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.

Load Mnist dataset..

Compile model..

Evaluate model on test Dataset
157/157 [==============================] - 4s 22ms/step - loss: 0.0417 - accuracy: 0.9913
loss: 0.042
acc: 0.991
```
### Finetuning
Here we just run finetuning once for demonstration. For further compiling we just used quantized_model.h5 generated before.

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > python finetune.py 

Load float model..

Load Mnist dataset..

Create quantize training model..
[INFO] Start CrossLayerEqualization...
10/10 [==============================] - 0s 33ms/step
[INFO] CrossLayerEqualization Done.

Compiling model..

Fit on Dataset..
Epoch 1/10
782/782 [==============================] - 48s 61ms/step - loss: 0.0077 - accuracy: 0.9978 - val_loss: 0.0738 - val_accuracy: 0.9882
Epoch 2/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0062 - accuracy: 0.9980 - val_loss: 0.0845 - val_accuracy: 0.9888
Epoch 3/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0058 - accuracy: 0.9983 - val_loss: 0.0810 - val_accuracy: 0.9885
Epoch 4/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0061 - accuracy: 0.9982 - val_loss: 0.0744 - val_accuracy: 0.9902
Epoch 5/10
782/782 [==============================] - 40s 51ms/step - loss: 0.0048 - accuracy: 0.9984 - val_loss: 0.0834 - val_accuracy: 0.9911
Epoch 6/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0047 - accuracy: 0.9986 - val_loss: 0.0807 - val_accuracy: 0.9893
Epoch 7/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0039 - accuracy: 0.9987 - val_loss: 0.0894 - val_accuracy: 0.9903
Epoch 8/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0034 - accuracy: 0.9989 - val_loss: 0.0863 - val_accuracy: 0.9904
Epoch 9/10
782/782 [==============================] - 39s 49ms/step - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.1043 - val_accuracy: 0.9893
Epoch 10/10
782/782 [==============================] - 39s 50ms/step - loss: 0.0044 - accuracy: 0.9986 - val_loss: 0.0994 - val_accuracy: 0.9908

Saved finetuned model as ./models/finetuned_model.h5

Evaluate model on test Dataset..
157/157 [==============================] - 1s 7ms/step - loss: 0.0675 - accuracy: 0.9920
loss: 0.068
acc: 0.992

```

### Compile into DPU model file

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > bash -x compile.sh 
+ ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
+ OUTDIR=./compiled_model
+ NET_NAME=customcnn
+ MODEL=./models/finetuned_model.h5
+ echo -----------------------------------------
-----------------------------------------
+ echo 'COMPILING MODEL FOR ZCU102..'
COMPILING MODEL FOR ZCU102..
+ echo -----------------------------------------
-----------------------------------------
+ compile
+ tee compile.log
+ vai_c_tensorflow2 --model ./models/finetuned_model.h5 --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json --output_dir ./compiled_model --net_name customcnn
/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.7/site-packages/xnnc/translator/tensorflow_translator.py:1843: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
  value = param.get(group).get(ds).value
[INFO] parse raw model     :100%|██████████| 10/10 [00:00<00:00, 16871.70it/s]               
[INFO] infer shape (NHWC)  :100%|██████████| 26/26 [00:00<00:00, 2956.30it/s]                
[INFO] generate xmodel     :100%|██████████| 26/26 [00:00<00:00, 5561.60it/s]                
[INFO] Namespace(inputs_shape=None, layout='NHWC', model_files=['./models/finetuned_model.h5'], model_type='tensorflow2', out_filename='./compiled_model/customcnn_org.xmodel', proto=None)
[INFO] tensorflow2 model: models/finetuned_model.h5
[OPT] No optimization method available for xir-level optimization.
[INFO] generate xmodel: /workspace/myproj/tf2-mnist-end-to-end/compiled_model/customcnn_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/vitis-ai-user/log/xcompiler-20210325-093926-3120"
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA0_B4096_MAX_BG2
[UNILOG][INFO] Graph name: mnist_customcnn_model, with op num: 42
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/customcnn.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 764142e83d074ea9470b9eb9d0757f68, and been saved to "/workspace/myproj/tf2-mnist-end-to-end/./compiled_model/md5sum.txt"
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MODEL COMPILED'
MODEL COMPILED
+ echo -----------------------------------------
-----------------------------------------

```

### Make target directory

```
(vitis-ai-tensorflow2) Vitis-AI /workspace/myproj/tf2-mnist-end-to-end > bash -x make_target.sh 
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET ZCU102 STARTED..'
MAKE TARGET ZCU102 STARTED..
+ echo -----------------------------------------
-----------------------------------------
+ TARGET_ZCU102=./target_zcu102
+ COMPILE_ZCU102=./compiled_model
+ APP=./application
+ NET_NAME=customcnn
+ rm -rf ./target_zcu102
+ mkdir -p ./target_zcu102/model_dir
+ cp ./application/app_mt.py ./target_zcu102
+ echo '  Copied application to TARGET_ZCU102 folder'
  Copied application to TARGET_ZCU102 folder
+ cp ./compiled_model/customcnn.xmodel ./target_zcu102/model_dir/.
+ echo '  Copied xmodel file(s) to TARGET_ZCU102 folder'
  Copied xmodel file(s) to TARGET_ZCU102 folder
+ mkdir -p ./target_zcu102/images
+ python generate_images.py --dataset=mnist --image_dir=./target_zcu102/images --image_format=jpg --max_images=10000
2021-03-25 09:42:34.445257: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Command line options:
 --dataset      :  mnist
 --subset       :  test
 --image_dir    :  ./target_zcu102/images
 --image_list   :  
 --label_list   :  
 --image_format :  jpg
 --max_images   :  10000
+ echo '  Copied images to TARGET_ZCU102 folder'
  Copied images to TARGET_ZCU102 folder
+ echo -----------------------------------------
-----------------------------------------
+ echo 'MAKE TARGET ZCU102 COMPLETED'
MAKE TARGET ZCU102 COMPLETED
+ echo -----------------------------------------
-----------------------------------------

```

## Run on zcu102
Refer to https://github.com/Xilinx/Vitis-AI/blob/master/setup/mpsoc/VART/README.md#step2-setup-the-target for board setup.
After that copy all the files in target_zcu102 directory to SD card.
Boot zcu102 from SD card.


