from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import ArcFaceModel,ArcFishStackModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf,generatePermKey

import modules.dataset as dataset

flags.DEFINE_string('cfg_path', './configs/arc_vgg16.yaml', 'config file path')
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

    basemodel = ArcFaceModel(backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=False,cfg=cfg)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        basemodel.load_weights(ckpt_path)
    else:
        print("[*] training from scratch.")
        sys.exit()

    model = ArcFishStackModel(basemodel=basemodel,backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True, cfg=cfg)
    model.summary(line_length=80)

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            inputs, labels = next(train_dataset)
            # print("********************")
            # print(inputs)
            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)
                # print(logist)

                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist)
                total_loss = pred_loss + reg_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy()))

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
        model.compile(optimizer=optimizer, loss=loss_fn)

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
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=epochs - 1)

    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
