import tensorflow as tf
from shutil import copy,rmtree,copytree,copy2
import os
import pathlib
import math
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import tqdm
import json

directory_str_tent = 'data/SESSION_TENT_NEW/SESSION1_LT'
directory_str_aqua = 'data/SESSION_AQUARIUM/SESSION1_LT'
ST_DIR_TENT = 'data/SESSION_TENT_NEW/SESSION1_ST/'
ST_DIR_AQUA = 'data/SESSION_AQUARIUM/SESSION1_ST/'

def _parse_tfrecord(binary_img=False, is_ccrop=False,cfg=None):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images(is_ccrop=is_ccrop,cfg=cfg)(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def _transform_images(is_ccrop=False,cfg=None):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (cfg['input_size_w']+20, cfg['input_size_h']+20))
        x_train = tf.image.random_crop(x_train, (cfg['input_size_w'], cfg['input_size_h'], 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_name, batch_size,
                          binary_img=False, shuffle=True, buffer_size=10240,
                          is_ccrop=False,cfg=None):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop,cfg=cfg),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



def addPrefix(path,prefix):
    for root, subdirs, files in os.walk(path):
        for name in files:
            curr_fld = os.path.basename(root)
            oldname = os.path.join(path, curr_fld, name)
            splt_name = name.split('.')
            myname = '_'.join([prefix, splt_name[0][-1], splt_name[0], curr_fld + '.' + splt_name[1]])
            newname = os.path.join(path, curr_fld, myname)
            os.rename(oldname, newname)
#args = parser.parse_args()
#byIDorByImages = args.byIDorByImages
#train_weight = args.train_weight
#print(byIDorByImages)
def generateDataset(byIDorByImages=True,train_weight=0.5,train_dir_tent='data/tmp_tent/train/',test_dir_tent='data/tmp_tent/test/',includeST=True, includeTentnAquaBoth=False):
    test_dir_tent = 'data/tmp_tent/test/'
    train_dir_aqua = 'data/tmp_aqua/train/'
    test_dir_aqua = 'data/tmp_aqua/test/'

    # remove any file exist
    if os.path.exists(train_dir_tent):
        rmtree(train_dir_tent)
        # rmtree(train_dir_aqua)
        rmtree(test_dir_tent)
        # rmtree(test_dir_aqua)

    # check_folder(train_dir)
    check_folder(test_dir_tent)
    check_folder(test_dir_aqua)

    # first copy ST to train
    if includeST:
        copytree(ST_DIR_TENT, train_dir_tent)
        pre = "tent_st"
        addPrefix(train_dir_tent, pre)


    SPLIT_WEIGHTS_INTRA_ID = (
        train_weight,1-train_weight, 0.0)  # train cv val vs test for each identity, 50% are taken as train and 50% as test
    SPLIT_WEIGHTS_INTER_ID = (
        train_weight, 1 - train_weight, 0.0)  # train cv val vs test for each identity, 50% are taken as train and 50% as test

    if byIDorByImages==1:
        train_ID_list, test_ID_list = splitID(directory_str_tent,SPLIT_WEIGHTS_INTRA_ID)
        print(train_ID_list)
        print("above IDs are used in training, remianing will be used for testing, in dir tmp_tent/test/*")

        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION1_LT', train_ID_list,train_dir_tent,test_dir_tent,pre='sess1')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION2', train_ID_list,train_dir_tent,test_dir_tent,pre='sess2')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION3', train_ID_list,train_dir_tent,test_dir_tent,pre='sess3')
        generateDatasetBySplitingIdentity('data/SESSION_TENT_NEW/SESSION4', train_ID_list,train_dir_tent,test_dir_tent,pre='sess4')


        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION1_LT', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess1')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION2', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess2')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION3', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess3')
        # generateDatasetBySplitingIdentity('data/SESSION_AQUARIUM/SESSION4', train_ID_list, train_dir_aqua, test_dir_aqua,pre='sess4')

    else:
        print('All IDs are used in training')




def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def generateDatasetBySplitingIdentity(directory_str,train_list,train_dir,test_dir,pre='sess1'):
    g = os.walk(directory_str)
    for path, dir_list, file_list in g:
        for id_dir in dir_list:
            if train_list.__contains__(id_dir): # in train set, maybe the filename will be same, hence we need to rename each session
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    #print(images)
                    dst_dir = os.path.join(train_dir, id_dir)
                    check_folder(dst_dir)
                    head, tail = os.path.split(images)
                    finalpath=os.path.join(dst_dir, pre+tail)
                    copy(images, finalpath)

            else:
                data_dir = pathlib.Path(os.path.join(path, id_dir))
                pic_list = list(data_dir.glob('*.png'))
                for images in pic_list:
                    dst_dir = os.path.join(test_dir,str.split(directory_str,'/')[-1], id_dir)
                    check_folder(dst_dir)
                    copy(images, dst_dir)


def splitID(directory_str,SPLIT_WEIGHTS_INTRA_ID):
    dir_list = [o for o in os.listdir(directory_str) if os.path.isdir(os.path.join(directory_str, o))]
    ids=len(dir_list)
    train_num = math.floor(ids * SPLIT_WEIGHTS_INTRA_ID[0])
    train_ID_list = random.sample(dir_list, k=train_num)
    test_ID_list=[]
    for anyid in dir_list:
        if not train_ID_list.__contains__(anyid):
            test_ID_list.append(anyid)
    return  train_ID_list,test_ID_list

def getfilelist(dirs):
  Filelist = []
  for home, dirs, files in os.walk(dirs):
    for filename in files:
# 文件名列表，包含完整路径
      Filelist.append(os.path.join(home, filename))
# # 文件名列表，只包含文件名
# Filelist.append( filename)
  return Filelist

def aug_data(orig_path,SAVE_PATH,num_aug_per_img=5):
    alllist = getfilelist(orig_path)
    num_imgs = len(alllist);
    print('total number of images:', num_imgs)

    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=False,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(alllist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=16, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if i > num_aug_per_img:
                break

def aug_data_sess1(orig_path,SAVE_PATH,k=1): # use k images from testing dataset as gallery and train the model into a classfication model for this ten IDs
    subfolders = [f.path for f in os.scandir(orig_path) if f.is_dir()]
    Filelist = []
    for dirs in subfolders:
        filename = random.choices(os.listdir(dirs), k=k)  # change dir name to whatever
        print(filename)
        for file in filename:
            Filelist.append(os.path.join(dirs, file))
    selected_Filelist = Filelist
    num_imgs = len(selected_Filelist)
    print('total number of images:', num_imgs)

    num_aug_per_img = 20
    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(selected_Filelist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if i > num_aug_per_img:
                break

def aug_data_sess(orig_path,SAVE_PATH,k=1):
    subfolders = [f.path for f in os.scandir(orig_path) if f.is_dir()]
    Filelist = []
    for dirs in subfolders:
        filename = random.choices(os.listdir(dirs), k=k)  # change dir name to whatever
        print(filename)
        for file in filename:
            Filelist.append(os.path.join(dirs, file))
    selected_Filelist = Filelist
    num_imgs = len(selected_Filelist)
    print('total number of images:', num_imgs)

    num_aug_per_img = 20
    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(selected_Filelist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if i > num_aug_per_img:
                break


def loadTestDS(test_data_dir = './data/tmp_tent/test/SESSION1_LT',BATCH_SIZE=64,cfg=None,LableDict=None):
    def get_label(file_path):
      parts = tf.strings.split(file_path, '/')
      # wh = LableDict[parts[-2]]
      return parts[-2]
    def _transform_images(is_ccrop=False, cfg=None):
        def transform_images(x_train):
            x_train = tf.image.resize(x_train, (cfg['input_size_w'], cfg['input_size_h']))
            x_train = tf.image.random_flip_left_right(x_train)
            x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
            x_train = tf.image.random_brightness(x_train, 0.4)
            x_train = x_train / 255
            return x_train
        return transform_images
    def process_path(file_path):
      label = get_label(file_path)
    # load the raw data from the file as a string
      image_encoded = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(image_encoded, channels=3)
      img = _transform_images(cfg=cfg)(img)
      return img, label
    list_ds = tf.data.Dataset.list_files(str(test_data_dir + '/*/*'))
    # print(f"we have total {self.image_count} images in this folder")
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset =labeled_ds.batch(BATCH_SIZE)
    return dataset


def loadTrainDS(test_data_dir = './data/tmp_tent/train/SESSION1_LT',BATCH_SIZE=64,cfg=None):
    def get_label(file_path):
      parts = tf.strings.split(file_path, '/')
      return parts[-2]
    def _transform_images(is_ccrop=False, cfg=None):
        def transform_images(x_train):
            x_train = tf.image.resize(x_train, (cfg['input_size_w'] + 20, cfg['input_size_h'] + 20))
            x_train = tf.image.random_crop(x_train, (cfg['input_size_w'], cfg['input_size_h'], 3))
            x_train = tf.image.random_flip_left_right(x_train)
            x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
            x_train = tf.image.random_brightness(x_train, 0.4)
            x_train = x_train / 255
            return x_train
        return transform_images

    def process_path(file_path):
      label = get_label(file_path)
    # load the raw data from the file as a string
      image_encoded = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(image_encoded, channels=3)
      img = _transform_images(cfg=cfg)(img)
      return img, label
    list_ds = tf.data.Dataset.list_files(str(test_data_dir + '*/*'))
    # print(f"we have total {self.image_count} images in this folder")
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
    dataset =labeled_ds.batch(BATCH_SIZE)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset