'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers

'''
Multi-scale Dilated Convolution module
 filters, kernel_size, strides=(1, 1), padding='valid'
'''

def MDCM(input_shape=None, name="Multi-scale-Dilated"):
    weight_decay = 0.005
    img_x = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(img_x)

    x1 = Conv2D(64, (3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x3 = Conv2D(64, (3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x4 = Conv2D(64, (3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    merge1 = concatenate([x1, x2, x3, x4])

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(merge1)

    x1 = Conv2D(256, (3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x2 = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(4, 4), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x3 = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(8, 8), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x4 = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(16, 16), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    merge2 = concatenate([x1, x2, x3, x4])

    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(merge2)

    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Flatten()(x)
    output = Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation='sigmoid')(x)
    model = Model(inputs=img_x, outputs=output, name=name)
    return model
