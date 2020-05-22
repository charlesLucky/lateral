'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


class LoadFishDataUtil():
    def __init__(self, directory_str, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES=None,
                 SPLIT_WEIGHTS=(0.7, 0.15, 0.15)):
        self.directory_str = directory_str
        self.SPLIT_WEIGHTS = SPLIT_WEIGHTS
        self.BATCH_SIZE = BATCH_SIZE
        self.data_dir = pathlib.Path(directory_str)
        self.image_count = len(list(self.data_dir.glob('*/*.png')))  # Opend
        if CLASS_NAMES is None:
            self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"])
        else:
            self.CLASS_NAMES = CLASS_NAMES

        self.class_num = len(self.CLASS_NAMES)

        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.STEPS_PER_EPOCH = np.ceil(self.image_count / BATCH_SIZE)

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')
        # The second to last is the class-directory
        # print(parts[-2] == self.CLASS_NAMES)
        wh = tf.where(tf.equal(self.CLASS_NAMES, parts[-2]))
        return parts[-2] == self.CLASS_NAMES

    def get_label_withname(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory

        wh = tf.where(tf.equal(self.CLASS_NAMES, parts[-2]))
        return wh

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # img = (img/127.5) - 1
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def process_path_resnet(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # img = (img/127.5) - 1
        # resize the image to the desired size.
        img = tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])
        img = preprocess_input(img)
        return img, label

    def process_path_withname(self, file_path):
        label = self.get_label_withname(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(self.BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def prepare_for_testing(self, ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        # ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        # ds = ds.repeat()

        ds = ds.batch(self.BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def loadFishData(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)

        train_size = int(self.SPLIT_WEIGHTS[0] * self.image_count)
        val_size = int(self.SPLIT_WEIGHTS[1] * self.image_count)
        test_size = int(self.SPLIT_WEIGHTS[2] * self.image_count)
        train_ds = self.prepare_for_training(self.labeled_ds)

        full_dataset = train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)
        train_dataset = full_dataset.take(train_size)
        remianing_train_set = full_dataset.skip(train_size)
        val_dataset = remianing_train_set.take(val_size)
        remain_val_set = remianing_train_set.skip(val_size)
        test_dataset = remain_val_set.take(test_size)
        return train_dataset, val_dataset, test_dataset, self.STEPS_PER_EPOCH, self.CLASS_NAMES, self.class_num

    def loadTestFishData(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        # print(f"we have total {self.image_count} images in this folder")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.labeled_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        dataset = self.labeled_ds.batch(self.BATCH_SIZE)  # 按照顺序取出4行数据，最后一次输出可能小于batch
        # dataset = dataset.repeat()#数据集重复了指定次数
        # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
        # 为了配合输出次数，一般默认repeat()空
        # test_ds = self.prepare_for_testing(self.labeled_ds)
        # test_ds = self.labeled_ds
        return dataset, self.class_num

    def loadTestFishData_resnet(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        # print(f"we have total {self.image_count} images in this folder")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.labeled_ds = list_ds.map(self.process_path_resnet, num_parallel_calls=self.AUTOTUNE)
        dataset = self.labeled_ds.batch(self.BATCH_SIZE)  # 按照顺序取出4行数据，最后一次输出可能小于batch
        # dataset = dataset.repeat()#数据集重复了指定次数
        # repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
        # 为了配合输出次数，一般默认repeat()空
        # test_ds = self.prepare_for_testing(self.labeled_ds)
        # test_ds = self.labeled_ds
        return dataset, self.class_num

    def loadFishDataWithname(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.labeled_ds = list_ds.map(self.process_path_withname, num_parallel_calls=self.AUTOTUNE)

        train_size = int(self.SPLIT_WEIGHTS[0] * self.image_count)
        val_size = int(self.SPLIT_WEIGHTS[1] * self.image_count)
        test_size = int(self.SPLIT_WEIGHTS[2] * self.image_count)
        train_ds = self.prepare_for_training(self.labeled_ds)

        full_dataset = train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=False)
        train_dataset = full_dataset.take(train_size)
        remianing_train_set = full_dataset.skip(train_size)
        val_dataset = remianing_train_set.take(val_size)
        remain_val_set = remianing_train_set.skip(val_size)
        test_dataset = remain_val_set.take(test_size)
        return train_dataset, val_dataset, test_dataset, self.STEPS_PER_EPOCH, self.CLASS_NAMES, self.class_num

    def loadTestFishDataWithname(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'))
        print(f"we have total {self.image_count} images in this folder")
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.labeled_ds = list_ds.map(self.process_path_withname, num_parallel_calls=self.AUTOTUNE)
        dataset = self.labeled_ds.batch(self.BATCH_SIZE)  # 按照顺序取出4行数据，最后一次输出可能小于batch

        return dataset, self.class_num

    #### imclearborder definition
