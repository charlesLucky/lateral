import  os
import  tensorflow as tf
import  numpy as np
from utils import LoadFishDataUtil,check_folder,calPerformance,traversalDir_FirstDirCount
import pathlib
import bob.measure
from matplotlib import pyplot
import h5py
from numpy.linalg import norm
from reportUtil import reportPerformance
import argparse


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7024)]) # where is my memory?
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

parser = argparse.ArgumentParser(description='Split the data and generate the train and test set')
parser.add_argument('modelckpt', help='the modelckpt', nargs='?', type=str)
parser.add_argument('plotDir', help='the plot save directory', nargs='?', type=str)

train_path = './tmp/'

IMG_WIDTH=320
IMG_HEIGHT=75
BATCH_SIZE = 8

SPLIT_WEIGHTS=(0.85, 0.15, 0.0)# train cv val vs test
class_num=traversalDir_FirstDirCount(train_path+'/train')
# taken from the train model
input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)
print('total class:',class_num)

FREEZE_LAYERS = 2  # freeze the first this many layers for training
# build model and optimizer
net = InceptionResNetV2(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=input_shape)
x = net.output
x = Dropout(0.5)(x)
x = Flatten()(x)
x= Dense(512, kernel_regularizer=regularizers.l2(0.0))(x)
x= layers.Activation('relu')(x)
x= layers.BatchNormalization()(x)
output_layer = Dense(class_num)(x)

model = models.Model(inputs=net.input, outputs=output_layer)


def getTestingEmbedding(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model):
    # print(CLASS_NAMES)
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess2_test_dataset, sess2_class_num = testloadData.loadTestFishData()
    test_num_elements = tf.data.experimental.cardinality(sess2_test_dataset).numpy()
    print(
        f"we have total {test_num_elements} batches of images for testing, around {test_num_elements*BATCH_SIZE} samples")
    # evaluate on test set
    # scores = model.evaluate(sess2_test_dataset, verbose=1)
    # print("Final test loss and accuracy :", scores)
    from numpy import linalg as LA

    feats = []
    names = []
    feature_model = models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization_203').output)
    n = 0
    for image_batch, label_batch in sess2_test_dataset:
        print(image_batch.shape)
        feature = feature_model(image_batch)

        # print(n)
        # print(feature.shape[0])
        for i in range(feature.shape[0]):
            n = n + 1
            feats.append(feature[i])
            names.append(np.argwhere(label_batch[i]).ravel())
            indxmax = np.argmax(feature[i])
            # print('predictions max index:',indxmax)
            # print('predictions:', CLASS_NAMES[indxmax] )
            # print('real:', CLASS_NAMES[np.argwhere(label_batch[i]).ravel()])

    print(f"finanly we have {n} samples extracted features")
    feats3 = np.array(feats)
    names3 = np.array(names)
    return feats3,names3



args = parser.parse_args()
plotDir = args.plotDir
modelckpt = args.modelckpt
if __name__ == "__main__":
    model.load_weights('model/'+modelckpt)
    print(model.summary())

    CLASS_NAMES = None
    test_data_dir = 'tmp/test/SESSION3'
    feats3, names3 = getTestingEmbedding(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES, model)

    test_data_dir = 'tmp/test/SESSION2'
    feats2, names2 = getTestingEmbedding(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES, model)

    test_data_dir = 'tmp/test/SESSION1_LT'
    feats1, names1 = getTestingEmbedding(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES, model)

    EER1vs2, gscores1vs2, iscores1vs2 = calPerformance(feats1, feats2, names1, names2)
    EER1vs3, gscores1vs3, iscores1vs3 = calPerformance(feats1, feats3, names1, names3)
    EER2vs3, gscores2vs3, iscores2vs3 = calPerformance(feats2, feats3, names2, names3)

    # we assume you have your negatives and positives already split
    npoints = 100
    bob.measure.plot.roc(iscores1vs2, gscores1vs2, npoints, color=(0, 0, 0), linestyle='-', label='inter-session 1 vs 2')
    bob.measure.plot.roc(iscores1vs3, gscores1vs3, npoints, color=(0, 0, 0), linestyle='-.', label='inter-session 1 vs 3')
    bob.measure.plot.roc(iscores2vs3, gscores2vs3, npoints, color=(0, 0, 0), linestyle=':', label='inter-session 2 vs 3')
    pyplot.xlabel('FPR (%)')
    pyplot.ylabel('FNR (%)')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()
    pyplot.savefig(plotDir+'intersession.svg')

    EER1vs1, gscores1vs1, iscores1vs1 = calPerformance(feats1, feats1, names1, names1)
    EER2vs2, gscores2vs2, iscores2vs2 = calPerformance(feats2, feats2, names2, names2)
    EER3vs3, gscores3vs3, iscores3vs3 = calPerformance(feats3, feats3, names3, names3)

    npoints = 100
    bob.measure.plot.roc(iscores1vs1, gscores1vs1, npoints, color=(0, 0, 0), linestyle='-', label='intra-session 1 vs 1')
    bob.measure.plot.roc(iscores2vs2, gscores2vs2, npoints, color=(0, 0, 0), linestyle='-.', label='intra-session 2 vs 2')
    bob.measure.plot.roc(iscores3vs3, gscores3vs3, npoints, color=(0, 0, 0), linestyle=':', label='intra-session 3 vs 3')
    pyplot.xlabel('FPR (%)')
    pyplot.ylabel('FNR (%)')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.show()
    pyplot.savefig(plotDir+'intrasession.svg')

    print(f"EER1vs2 EER1vs3 EER2vs3: {EER1vs2} {EER1vs3} {EER2vs3}")
    print(f"EER1vs1 EER2vs2 EER3vs3: {EER1vs1} {EER2vs2} {EER3vs3}")

    reportPerformance(feats1, feats2, names1, names2, label='1vs2')
    reportPerformance(feats1, feats3, names1, names3, label='1vs3')
    reportPerformance(feats2, feats3, names2, names3, label='2vs3')

    reportPerformance(feats1, feats1, names1, names1, label='1vs1')
    reportPerformance(feats2, feats2, names2, names2, label='2vs2')
    reportPerformance(feats3, feats3, names3, names3, label='3vs3')


