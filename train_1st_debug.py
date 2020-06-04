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
        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']

        train_dataset = loadTrainDS('./data/tmp_tent/train/', BATCH_SIZE=cfg['batch_size'], cfg=cfg)

    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1


    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    # loss_fn = SoftmaxLoss()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    model.compile(optimizer=optimizer, loss=loss_fn,metrics=['accuracy'])
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
