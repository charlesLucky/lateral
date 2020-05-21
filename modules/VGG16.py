from tensorflow.keras import datasets, layers, optimizers, models, regularizers


def VGG16(input_shape=None, class_num=10, useAMSoftmax=False):
    weight_decay = 0.000

    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(class_num))
    # model.add(layers.Activation('softmax'))
    return model


def VGG16Dilated(input_shape=None, class_num=10, name="model", useAMSoftmax=False):
    weight_decay = 0.000
    model = models.Sequential(name=name)

    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))  # dilated
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, (3, 3), dilation_rate=(2, 2), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), dilation_rate=(4, 4), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), dilation_rate=(4, 4), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), dilation_rate=(4, 4), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(8, 8), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(8, 8), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(8, 8), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(16, 16), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(16, 16), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), dilation_rate=(16, 16), padding='same',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(class_num))
    # model.add(layers.Activation('softmax'))
    return model

