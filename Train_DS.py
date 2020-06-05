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

from modules.models import ArcFaceModel,FishModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf,generatePermKey

from modules.dataset  import loadTrainDS

from modules.LoadFishDataUtil import LoadFishDataUtil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import tqdm
import pathlib
from shutil import copy,rmtree,copytree,copy2

flags.DEFINE_string('cfg_path', './configs/ResNet50_1st.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True,cfg=cfg)
    model.summary(line_length=80)

    if cfg['train_dataset']:
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
                    copy(images, dst_dir)

                cnt = cnt + 1

        ds_path = './data/tmp_tent/train/'
        save_dir = './data/tmp_tent/train_ds/'
        # renameDir(ds_path, save_dir)

        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']

        train_dataset = loadTrainDS(save_dir, BATCH_SIZE=cfg['batch_size'], cfg=cfg)

    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1


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

        while epochs <= cfg['epochs']:
            inputs, labels = next(train_dataset)

            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)
                # print(logist)
                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist)* 10
                total_loss = pred_loss + reg_loss
                output = tf.argmax(tf.transpose(logist))
                correct = tf.shape(tf.where([output == labels]))[0] / cfg['batch_size']

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}, acc={:.4f}"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy(), correct.numpy()))

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                    cfg['sub_name'], epochs, steps % steps_per_epoch))

            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        mc_callback = ModelCheckpoint(
            'checkpoints/' + cfg['sub_name'] + '/e_{epoch}_b_{batch}.ckpt',
            save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
            save_weights_only=True)

        tb_callback = TensorBoard(log_dir='logs/',
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        callbacks = [mc_callback, tb_callback]

        model.fit(train_dataset,
                  epochs=cfg['epochs'],
                  callbacks=callbacks,
                  steps_per_epoch=10000,
                  initial_epoch=epochs - 1)

    print("[*] training done!")



if __name__ == '__main__':
    app.run(main)