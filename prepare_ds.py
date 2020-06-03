'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf
from modules.dataset import generateDataset, aug_data, aug_data_sess1, aug_data_sess
import json
from shutil import copy, rmtree, copytree, copy2

flags.DEFINE_string('dataset_path', './data/tmp_tent/SESSION1_ST_AUGMENT',
                    'path to dataset')
flags.DEFINE_string('output_path', './data/New_ROI_STLT_bin.tfrecord',
                    'path to ouput tfrecord')

flags.DEFINE_string('stage', '0',
                    'which stage,1,2')


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def labelToDigitDict(dataset_path):
        # below we code the label into consistent number start from 0
        ids_list = [o for o in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, o))]
        cnt = 0
        Label_dict = {}
        for id_str in ids_list:
            Label_dict[id_str] = cnt
            cnt = cnt + 1
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
        TRAIN_SAVE_PATH = './data/tmp_tent/test/SESSION_LT_AUGMENT'
        if os.path.exists(TRAIN_SAVE_PATH):
            rmtree(TRAIN_SAVE_PATH)
        aug_data_sess1('./data/tmp_tent/test/SESSION1_LT', TRAIN_SAVE_PATH, k=0)  # augmentation
        dataset_path = TRAIN_SAVE_PATH
        output_path = './data/New_ROI_LT1_bin.tfrecord'
        Label_dict = labelToDigitDict(dataset_path)
        with open('data/Label_dict_2ed.json', 'w') as fp:
            json.dump(Label_dict, fp, sort_keys=True, indent=4)
    elif FLAGS.stage == '3':
        TRAIN_SAVE_PATH = './data/tmp_tent/test/SESSION_LT_AUGMENT'
        if os.path.exists(TRAIN_SAVE_PATH):
            rmtree(TRAIN_SAVE_PATH)
        aug_data_sess('./data/tmp_tent/test/SESSION2', TRAIN_SAVE_PATH, k=0)
    elif FLAGS.stage == '4':
        TRAIN_SAVE_PATH = './data/tmp_tent/test/SESSION_LT_AUGMENT'
        aug_data_sess('./data/tmp_tent/test/SESSION3', TRAIN_SAVE_PATH, k=0)
    else:
        print('[*] stage should be given!')

    ####################################################################################


if __name__ == '__main__':
    try:
        # generateDataset(byIDorByImages=True, train_weight=0.67)  # half as train and half as test  0.67-> 20 as train 10 as test
        app.run(main)
    except SystemExit:
        pass
