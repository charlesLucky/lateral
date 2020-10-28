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
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import ArcFaceModel, FishModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf, generatePermKey

from modules.dataset import loadTrainDS
from PIL import Image
from modules.LoadFishDataUtil import LoadFishDataUtil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import tqdm
import pathlib
from shutil import copy, rmtree, copytree, copy2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imsave

flags.DEFINE_string('cfg_path', './configs/ResNet50_1st.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('rename', 0, 'rename all the file name to number format')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    batch_size = FLAGS.batch_size
    cfg['batch_size'] = batch_size
    model = ArcFaceModel(backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True, cfg=cfg,name=cfg['backbone_type'])
    model.summary(line_length=80)

    def check_folder(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name

    def renameDir(ds_path, save_dir):
        if os.path.exists(save_dir):
            rmtree(save_dir)

        dir_list = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]
        cnt = 0;
        for dir_name in tqdm.tqdm(dir_list):
            data_dir = pathlib.Path(os.path.join(ds_path, dir_name))
            pic_list = list(data_dir.glob('*.png'))

            for images in pic_list:
                dst_dir = os.path.join(save_dir, "%05d" % cnt)
                check_folder(dst_dir)
                img = mpimg.imread(images)
                # gray = rgb2gray(img)
                # image = cv2.imread(images)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # scale_ratio = 0.5
                # img_resized = cv2.resize(gray, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC)
                # image = color.rgb2gray(data.astronaut())
                # print(img.shape[1]) # 图片的尺寸

                img = img[:, round(img.shape[1] / 2):, :]

                image_resized = resize(img, (320, 320),
                                       anti_aliasing=True)
                # print(images)
                head_tail = os.path.split(images)
                imsave(dst_dir + '/' + head_tail[1], image_resized)
                # copy(images, dst_dir)

            cnt = cnt + 1

    ds_path = './data/tmp_tent/train/'
    save_dir = './data/tmp_tent/train_ds/'
    if FLAGS.rename:
        renameDir(ds_path, save_dir)

    logging.info("load fish training dataset.")
    dataset_len = cfg['num_samples']
    steps_per_epoch = dataset_len // batch_size

    train_dataset = loadTrainDS(save_dir, BATCH_SIZE=batch_size, cfg=cfg)

    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()
    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)
        es_cnt=0
        while epochs <= cfg['epochs']:
            if steps % 5 == 0:
                start = time.time()
            inputs, labels = next(train_dataset)
            # i = Image.fromarray(inputs[0,].numpy(), "RGB")
            # i.show()
            # print(inputs[0,].shape)
            # sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')
            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)

                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist) * 10
                total_loss = pred_loss + reg_loss
                output = tf.argmax(tf.transpose(logist))
                correct = tf.shape(tf.where([output == labels]))[0] / batch_size

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                end = time.time()
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}, acc={:.4f},time/step={:.2f}s, remaining-epoch={:.2f}min"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy(), correct.numpy(),end - start,(steps_per_epoch -(steps % steps_per_epoch)) * (end - start) /60.0))

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'loss/acc', correct, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)
            # if steps % cfg['save_steps'] == 0:
            #     print('[*] save ckpt file!')
            #     model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
            #         cfg['sub_name'], epochs, steps % steps_per_epoch))
            # if steps % steps_per_epoch == 0:
                # print('[*] save ckpt file!')
                # model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                #     cfg['sub_name'], epochs, steps % steps_per_epoch))
            if steps % steps_per_epoch == 0:
                if correct>=0.9988:
                    es_cnt = es_cnt+1
                if es_cnt>5:
                    break;
            steps += 1
            epochs = steps // steps_per_epoch + 1

    print("[*] training done!")
    model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
        cfg['sub_name'], epochs, steps % steps_per_epoch))


if __name__ == '__main__':
    app.run(main)
