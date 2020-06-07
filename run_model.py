import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('saved_model/model_10')

model.summary()

(_,_), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.reshape(-1, 28,28, 1)

x_test = x_test.astype('float32')
x_test = x_test / 255
y_test = to_categorical(y_test)

loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print('Restored model accuracy: {:5.2f}%'.format(100*acc))
