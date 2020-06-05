from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping

from modules.models import ArcFaceModel,ArcFishStackModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf,generatePermKey
from modules.LoadFishDataUtil import LoadFishDataUtil
from modules.evaluations import reportAccu
from modules.dataset import aug_data_sess,aug_data_sess1

import modules.dataset as dataset

flags.DEFINE_string('cfg_path', './configs/ResNet50_2ed_stage.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('epochs', '1', 'which epoch to start')
flags.DEFINE_string('stage', '2', 'which stage to start')
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

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['1st_sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        basemodel.load_weights(ckpt_path)
    else:
        print("[*] training from scratch.")


    logging.info("load fish LT sessions dataset.")

    CLASS_NAMES = None
    SPLIT_WEIGHTS = (0.9, 0.1, 0.0)  # train cv val vs test
    myloadData = LoadFishDataUtil('./data/tmp_tent/test/SESSION_LT_AUGMENT', cfg['batch_size'], cfg['input_size_w'], cfg['input_size_h'],
                                  CLASS_NAMES, SPLIT_WEIGHTS)
    train_dataset, val_dataset, test_dataset, STEPS_PER_EPOCH, CLASS_NAMES, class_num = myloadData.loadFishData()
    print(f'total class:{class_num}')

    epochs, steps = int(FLAGS.epochs), 1

    model = ArcFishStackModel(basemodel=basemodel,
                         num_classes=class_num,
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True, cfg=cfg)

    # FREEZE_LAYERS = 145
    for layer in model.layers:
        print('*1:',layer.name)
        if layer.name == 'arcface_model':
            layer.trainable = False

    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)

    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    # loss_fn = SoftmaxLoss()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn,metrics=['accuracy'])

    mc_callback = ModelCheckpoint(
        'checkpoints/' + cfg['sub_name'] + '/e_{epoch}.ckpt',#save_freq=cfg['save_steps'] * cfg['batch_size'],
         verbose=1, monitor='loss', save_best_only=True,
        save_weights_only=True)

    tb_callback = TensorBoard(log_dir='logs/',
                              update_freq=cfg['batch_size'] * 5,
                              profile_batch=0)
    tb_callback._total_batches_seen = steps
    tb_callback._samples_seen = steps * cfg['batch_size']

    es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=3)

    callbacks = [mc_callback, tb_callback,es]

    model.fit(train_dataset,
              epochs=cfg['epochs'],
              callbacks=callbacks,
              initial_epoch=epochs - 1)
    print("[*] training done!")

    File_log_name = 'logs/multistage_Ids10Test_tent_vote.log'
    scores_session1, scores_session2, scores_session3, scores_session4 =  reportAccu( cfg['batch_size'], cfg['input_size_w'],
                                  cfg['input_size_h'], CLASS_NAMES, model)
    printstr = f"stage: {FLAGS.stage}: {scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n"
    print(printstr)
    with open(File_log_name, encoding="utf-8", mode="a") as data:
        data.write(printstr)


if __name__ == '__main__':
    app.run(main)
