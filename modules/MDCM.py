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

'''
Multi-scale Dilated Convolution module
 filters, kernel_size, strides=(1, 1), padding='valid'
'''

def getMDCM(input_shape=None, name="Multi-scale-Dilated"):
    weight_decay = 5e-4
    img_x = Input(shape=input_shape)
    kernal_size = (3,3)
    # kernal_size = (5,3)

    x = Conv2D(16, kernal_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(img_x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x1 = Conv2D(16, kernal_size, dilation_rate=(1, 1), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x2 = Conv2D(16, kernal_size, dilation_rate=(2, 2), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x3 = Conv2D(16, kernal_size, dilation_rate=(4, 4), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x4 = Conv2D(16, kernal_size, dilation_rate=(8, 8), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    merge1 = concatenate([x1, x2, x3, x4])

    x = Conv2D(32, kernal_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(merge1)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x11 = Conv2D(32, kernal_size, dilation_rate=(1, 1), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x21 = Conv2D(32, kernal_size, dilation_rate=(2, 2), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x31 = Conv2D(32, kernal_size, dilation_rate=(4, 4), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x41 = Conv2D(32, kernal_size, dilation_rate=(8, 8), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    merge2 = concatenate([x11, x21, x31, x41])
    # x = BatchNormalization()(merge2)
    # x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(64, kernal_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(merge2)


    x12 = Conv2D(64, kernal_size, dilation_rate=(1, 1), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x22 = Conv2D(64, kernal_size, dilation_rate=(2, 2), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x32 = Conv2D(64, kernal_size, dilation_rate=(4, 4), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x42 = Conv2D(64, kernal_size, dilation_rate=(8, 8), padding='same', activation='relu',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    merge3 = concatenate([x12, x22, x32, x42])

    # x = Conv2D(128, kernal_size, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(merge3)
    #
    #
    # x12 = Conv2D(128, kernal_size, dilation_rate=(1, 1), padding='same', activation='relu',
    #             kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x22 = Conv2D(128, kernal_size, dilation_rate=(2, 2), padding='same', activation='relu',
    #             kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x32 = Conv2D(128, kernal_size, dilation_rate=(4, 4), padding='same', activation='relu',
    #             kernel_regularizer=regularizers.l2(weight_decay))(x)
    # x42 = Conv2D(128, kernal_size, dilation_rate=(8, 8), padding='same', activation='relu',
    #             kernel_regularizer=regularizers.l2(weight_decay))(x)
    #
    # merge4 = concatenate([x12, x22, x32, x42])


    x = Conv2D(128, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(merge3)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(128, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, kernal_size, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, kernal_size, padding='same', strides=(2, 2), activation='relu',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    output = Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation='sigmoid')(x)
    model = Model(inputs=img_x, outputs=output, name=name)
    print(model.summary())
    return model

class MDCM(keras.layers.Layer):

  def __init__(self, input_shape=(112,112,3)):
    super(MDCM, self).__init__()
    self.model = getMDCM(input_shape)

  def call(self, inputs):
    return self.model(inputs)
