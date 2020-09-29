'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers


def Localnet():
    """LocalNet Model"""
    def localnet(x_in):
        weight_decay = 5e-4

        x = Conv2D(24, 5, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x_in)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Conv2D(32, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Conv2D(48, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Conv2D(64, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

        x = Flatten()(x)

        x = Dense(64, kernel_regularizer=regularizers.l2(weight_decay))(x)
        output = Dense(3, kernel_regularizer=regularizers.l2(weight_decay))(x)

        return output
    return localnet
#
#
# def getNetwork(input_shape=None, batch_size = 20, name="LocalNetwork"):
#     weight_decay = 5e-4
#     x = inputs = Input(shape=input_shape,name='input_image')
#
#     x = Conv2D(24,5, stride = 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = MaxPooling2D(pool_size=(2, 2), stride = 2)(x)
#
#     x = Conv2D(32,3, stride = 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = MaxPooling2D(pool_size=(2, 2), stride = 2)(x)
#
#     x = Conv2D(48, 3, stride=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = MaxPooling2D(pool_size=(2, 2), stride=2)(x)
#
#     x = Conv2D(64, 3, stride=1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = MaxPooling2D(pool_size=(2, 2), stride=2)(x)
#
#     x = Flatten()(x)
#
#     x = Dense(64, kernel_regularizer=regularizers.l2(weight_decay))(x)
#     output = Dense(3, kernel_regularizer=regularizers.l2(weight_decay))(x)
#
#     model = Model(inputs=inputs, outputs=output, name=name)
#     print(model.summary())
#     return model
#
# class LocalNet(keras.layers.Layer):
#
#   def __init__(self, input_shape=(112,112,3),batch_size=20):
#     super(LocalNet, self).__init__()
#     self.model = getNetwork(input_shape, batch_size=batch_size)
#
#   def call(self, inputs):
#     return self.model(inputs)
