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

from modules.models import ArcFaceModel,ArcFishStackModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf2,generatePermKey,get_ckpt_inf3

from modules.dataset  import loadTrainDS
from modules.evaluations import reportAccu_ds
import tqdm
import pathlib
from shutil import copy,rmtree,copytree,copy2

flags.DEFINE_string('cfg_path', './configs/ResNet50_1st.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')
flags.DEFINE_integer('stage', 2, 'which stage to start')
flags.DEFINE_integer('batch_size', 64, 'batch size')


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    batch_size = FLAGS.batch_size

    basemodel = ArcFaceModel(backbone_type=cfg['backbone_type'],
                             num_classes=cfg['num_classes'],
                             head_type=cfg['head_type'],
                             embd_shape=cfg['embd_shape'],
                             w_decay=cfg['w_decay'],
                             training=False, cfg=cfg)
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['1st_sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        basemodel.load_weights(ckpt_path)
    else:
        print("[*] training from scratch.")
    model = ArcFishStackModel(basemodel=basemodel,
                              num_classes=10,
                              head_type=cfg['head_type'],
                              embd_shape=cfg['embd_shape'],
                              w_decay=cfg['w_decay'],
                              training=True, cfg=cfg)
    # FREEZE_LAYERS = 145
    # for layer in model.layers:
    #     if layer.name == 'arcface_model':
    #         layer.trainable = False

    ckpt_path = tf.train.latest_checkpoint('./checkpoints222/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf3(ckpt_path)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)

    def renameDir(ds_path, save_dir):
        def check_folder(dir_name):
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            return dir_name

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

    # renameDir(ds_path, save_dir)
    # renameDir('./data/tmp_tent/test/SESSION2', save_dir)

    # renameDir('./data/tmp_tent/test/SESSION1_LT', './data/stage2/SESSION1_LT/')
    renameDir('./data/tmp_tent/test/SESSION_LT_AUGMENT', './data/stage2/SESSION1_LT/')
    renameDir('./data/tmp_tent/test/SESSION2', './data/stage2/SESSION2/')
    renameDir('./data/tmp_tent/test/SESSION3', './data/stage2/SESSION3/')
    renameDir('./data/tmp_tent/test/SESSION4', './data/stage2/SESSION4/')

    if FLAGS.stage == 2:
        #'./data/tmp_tent/test/SESSION_LT_AUGMENT'
        save_dir = './data/stage2/SESSION1_LT/'
    elif FLAGS.stage==3:
        save_dir = './data/stage2/SESSION2/'
    elif FLAGS.stage==4:
        save_dir = './data/stage2/SESSION3/'

    logging.info("load dataset."+save_dir)


    train_dataset = loadTrainDS(save_dir, BATCH_SIZE=batch_size, cfg=cfg)
    num_elements = tf.data.experimental.cardinality(train_dataset).numpy()
    # dataset_len = cfg['num_samples']
    dataset_len = num_elements
    steps_per_epoch = dataset_len // batch_size

    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()
    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

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
                correct = tf.shape(tf.where([output == labels]))[0] / batch_size

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

    print("[*] training done!")
    model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
        cfg['sub_name'], epochs, steps % steps_per_epoch))

    File_log_name = 'logs/multistage_Ids10Test_tent_vote2.log'
    scores_session1, scores_session2, scores_session3, scores_session4 =  reportAccu_ds(cfg, model)
    printstr = f"stage:{cfg['backbone_type']} {FLAGS.stage} {scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n"
    print(printstr)
    with open(File_log_name, encoding="utf-8", mode="a") as data:
        data.write(printstr)

if __name__ == '__main__':
    app.run(main)
