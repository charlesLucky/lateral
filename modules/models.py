import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    GRU,
    SimpleRNN,
    Embedding,
    concatenate
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    ResNet101,
    InceptionResNetV2,
    InceptionV3,
    Xception,
    VGG16,
    VGG19
)
from modules.MDCM import MDCM
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50', use_pretrain=True,batch_size=None):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'InceptionResNetV2':
            return InceptionResNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'InceptionV3':
            return InceptionV3(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'Xception':
            return Xception(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'VGG16':
            return VGG16(input_shape=x_in.shape[1:], include_top=False,
                         weights=weights)(x_in)
        elif backbone_type == 'VGG19':
            return VGG19(input_shape=x_in.shape[1:], include_top=False,
                         weights=weights)(x_in)
        elif backbone_type == 'MDCM':
            return MDCM(input_shape=x_in.shape[1:], kernal_size=(3, 3))(x_in)
        elif backbone_type == 'MDCMrect':
            return MDCM(input_shape=x_in.shape[1:], kernal_size=(3, 5))(x_in)
        elif backbone_type == 'MDCMHOG':
            return MDCM(input_shape=x_in.shape[1:], kernal_size=(3, 3),ifHOG = True,batch_size=batch_size)(x_in)
        else:
            raise TypeError('backbone_type error!')


    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""

    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)

    return output_layer


def OutputLayerRNN(embd_shape, w_decay=5e-4, name='OutputLayerRNN'):
    """Output Later"""

    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        model = tf.keras.Sequential()
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Embedding(input_dim=512, output_dim=512) (x)
        x = GRU(512, return_sequences=False)(x)
        # x = SimpleRNN(256)(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)

    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""

    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))

    return arc_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""

    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x) #, activation='sigmoid'
        return Model(inputs, x, name=name)(x_in)

    return norm_head


def ArcFaceModel(channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False, cfg=None):
    """Arc Face Model"""
    x = inputs = Input([cfg['input_size_w'], cfg['input_size_h'], channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain,batch_size=cfg['batch_size'])(x)

    if cfg['rnn']:
        embds1 = OutputLayer(embd_shape, w_decay=w_decay)(x)
        embds2 = OutputLayerRNN(embd_shape, w_decay=w_decay)(x)
        embds = concatenate([embds1, embds2])
    else:
        embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model(inputs, logist, name=name)
    else:
        return Model(inputs, embds, name=name)



def FishModel(channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False, cfg=None):
    """Arc Face Model"""
    x = inputs = Input([cfg['input_size_w'], cfg['input_size_h'], channels], name='input_image')
    backbonemodel = Model(inputs, Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain,batch_size=cfg['batch_size'])(x), name="backbone1")
    backbonemodel2 = Model(inputs, Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain,batch_size=cfg['batch_size'])(x), name="backbone2")
    x1 = backbonemodel(x[:,:,:round(cfg['input_size_h']/2),:])
    x2 = backbonemodel2(x[:,:,round(cfg['input_size_h']/2):,:])
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x = concatenate([x1,x2])

    if cfg['rnn']:
        embds1 = OutputLayer(embd_shape, w_decay=w_decay)(x)
        embds2 = OutputLayerRNN(embd_shape, w_decay=w_decay)(x)
        embds = concatenate([embds1, embds2])
    else:
        embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model(inputs, logist, name=name)
    else:
        return Model(inputs, embds, name=name)


# def FishModel(channels=3, num_classes=None, name='fish_model',
#               margin=0.5, logist_scale=64, embd_shape=512,
#               head_type='ArcHead', backbone_type='ResNet50',
#               w_decay=5e-4, use_pretrain=True, training=False, cfg=None):
#     """Arc Face Model"""
#     x = inputs = Input([cfg['input_size_w'], cfg['input_size_h'], channels], name='input_image')
#     x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain,batch_size=cfg['batch_size'])(x)
#     embds = OutputLayer(embd_shape, w_decay=w_decay)(x)
#     logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
#     return Model(inputs, logist, name=name)


def ArcFishStackModel(basemodel=None, channels=3, num_classes=None, name='arcface_model',
                      margin=0.5, logist_scale=64, embd_shape=512,
                      head_type='ArcHead',
                      w_decay=5e-4, use_pretrain=True, training=False, cfg=None):
    """Arc Face Model"""
    x = inputs = Input([cfg['input_size_w'], cfg['input_size_h'], channels], name='input_image')
    embds = basemodel(x)
    assert num_classes is not None
    x = Dense(512, kernel_regularizer=_regularizer(w_decay), activation='relu')(embds)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(256, kernel_regularizer=_regularizer(w_decay))(x)
    logist = NormHead(num_classes=num_classes, w_decay=w_decay)(x)
    return Model(inputs, logist, name=name)
