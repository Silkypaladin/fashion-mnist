# Fashion-MNIST classification problem

## Introduction

FashionMnist is a data set consisting of 70,000 examples of Zalando's article images - pieces of clothing. Each image is associated with one label from 10 classes.
The goal is to correctly identify each fashion product from given dataset and maximize the accuracy of predictions of the model, using different Machine Learning Algorithms to achieve that.

## Methods

To maximize the accuracy, I decided to build a Convolutional Neural Network with several layers. First, I normalized the input data, so every piece is on the same scale. It helps reducing the difficulty of the problem.
```python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255
```
Next, I transformed the labels to a binary format.
```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```
I used a sequential model with batch normalization, as it reduces the amount by what the hidden values shift around (reduces overfitting and has slight regularization effect) and added the layers as follows: 

**Conv2D** - A 2D Convolutional layer with 128 filters, 4x4 kernel, Relu activation function to speed up the training process
**MaxPooling** - A 2x2 maximum pooling layer to reduce computational cost and create a summarized version of the features, small changes in the input image won't modify the pooled output too much ([Source](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/))
**Dropout** - A dropout of rate 0.3 - randomly ignoring units with probability of p to reduce overfitting

**Conv2D** - A 2D Convolutional layer with 64 filters, 3x3 kernel, Relu activation function to speed up the training process
**MaxPooling** - A 2x2 maximum pooling layer
**Dropout** - A dropout of rate 0.3

**Flatten** - A flatten operation performed on the feature map ([More info here](https://www.superdatascience.com/convolutional-neural-networks-cnn-step-3-flattening/))

**Dense** - a fully-connected layer with 256 units and Relu activation function
**Dense** - a fully-connected layer with 64 units
**Dense** - a final, fully connected layer with units equal to 10 (number of classes in the dataset) and softmax activation function (useful in multiclass classification) when the input can be of only one class)
I used *Nadam* optimizer, as it provided slightly better accuracy than *Adam*, batch size of 96 and 10 epochs.

The pages given below were a massive help in building my own neural network.
[Why use batch normalization](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
[Helpful CNN tutorial](https://medium.com/datadriveninvestor/implementing-convolutional-neural-network-using-tensorflow-for-fashion-mnist-caa99e423371)
[Useful step-by-step guide to build a neural network](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-i-hyper-parameter-8129009f131b)


## Results

I managed to achieve around 92% accuracy and around 23% test loss, tested on 0.2 and 0.3 dropout rate. The table below compares some of the results provided in **Benchmark** section [here](https://github.com/zalandoresearch/fashion-mnist) with my result.

| Classifier     | Preprocessing | Fashion test accuracy | Submitter         |
|----------------|---------------|-----------------------|-------------------|
| 2 Conv+pooling | None          | 0.876                 | Khasif Rasoul     |
| 2 Conv+pooling | None          | 0.916                 | Tensorflow's doc  |
| 2 Conv+pooling | Normalization | 0.925                 | Silkypaladin (Me) |

## Usage

To run the code, several libraries need to be installed.

Tensorflow - The best way to install it is to follow the official tutorial [here](https://www.tensorflow.org/install/pip)
Keras - type in the following command
```bash
pip install keras
```

For additional, helpful functions install numpy and matplotlib.

```bash
pip install numpy
pip install matplotlib
```

The dataset is loaded automatically, via tensorflow pipeline. Trained model is saved in */saved_model* directory.
To train the model yourself, open a command line in the project directory and type in the following command:
```bash
python recognition.py
```
To use the trained model, run the command below:
```bash
python run_model.py
```


