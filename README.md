# SpeechRecognitionHuaweiNPU
Training a speech recogntion system with tensorflow 1.3 that is compatible with the Neural Processing Uni on Huaweis latest smartphones.

In Script.zip one can find the final documentation of our project.

# How to use the repo
This is the code release for Huawei TechChallenge. This code trains a speech recognition network that is compatible with the NPU on latest Huawei devices. In order to get started please download librispeech (http://www.openslr.org/12/) or any other speech data set and perform preprocessing similar to the example in Preprocess_Data.ipynb. After one has the correctly preprocessed data in the corresponding folder structure, one is able to follow Huawei_compatible_model.ipynb.

# Architecture
Due to the beta character of of NPU we were only able to use a limited amout of tensorflow operations. We ended up using following architecture as it performed well enough for the first version of our app. First we apply 1D convolution simulated by a time-distributed dense layer. Then we a have residual connection bypassing a one directional LSTM. The filnal layer is the time distributed dense layer with softmax activation. 
