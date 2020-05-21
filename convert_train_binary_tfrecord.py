from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf
from modules.dataset import generateDataset, aug_data,aug_data_sess1
import json



flags.DEFINE_string('dataset_path', './data/tmp_tent/SESSION1_ST_AUGMENT',
                    'path to dataset')
flags.DEFINE_string('output_path', './data/New_ROI_STLT_bin.tfrecord',
                    'path to ouput tfrecord')

flags.DEFINE_string('stage', '0',
                    'which stage,1,2')

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    def labelToDigitDict(dataset_path):
        # below we code the label into consistent number start from 0
        ids_list = [o for o in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, o))]
        cnt=0
        Label_dict = {}
        for id_str in ids_list:
            Label_dict[id_str] = cnt
            cnt = cnt+1
        return Label_dict

    dataset_path = ''
    output_path = ''
    if FLAGS.stage == '1':
        generateDataset(byIDorByImages=True,
                        train_weight=0.67)  # half as train and half as test  0.67-> 20 as train 10 as test

        orig_path = './data/tmp_tent/train/'
        SAVE_PATH = './data/tmp_tent/SESSION1_ST_AUGMENT'

        aug_data(orig_path, SAVE_PATH, num_aug_per_img=5)
        dataset_path = SAVE_PATH
        output_path = './data/New_ROI_STLT_bin.tfrecord'
        Label_dict = labelToDigitDict(dataset_path)
    elif FLAGS.stage == '2':
        TRAIN_SAVE_PATH = './data/tmp_tent/test/SESSION_LT_AUGMENT'
        aug_data_sess1('./data/tmp_tent/test/SESSION1_LT', TRAIN_SAVE_PATH,k=2)  # augmentation
        dataset_path = TRAIN_SAVE_PATH
        output_path = './data/New_ROI_LT1_bin.tfrecord'
        Label_dict = labelToDigitDict(dataset_path)
        with open('data/Label_dict_2ed.json', 'w') as fp:
            json.dump(Label_dict, fp, sort_keys=True, indent=4)
    else:
        print('[*] stage should be given!')
        

    ####################################################################################

    # dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    samples = []
    logging.info('Reading data list...')
    for id_name in tqdm.tqdm(os.listdir(dataset_path)):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.png'))
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, id_name, filename))
    random.shuffle(samples)

    logging.info('Writing tfrecord file...')
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=Label_dict[id_name],
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    try:
        generateDataset(byIDorByImages=True, train_weight=0.67)  # half as train and half as test  0.67-> 20 as train 10 as test
        # app.run(main)
    except SystemExit:
        pass
