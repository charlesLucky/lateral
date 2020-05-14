from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import tensorflow as tf

from modules.evaluations import reportAccu
from modules.models import ArcFaceModel,ArcFishStackModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

flags.DEFINE_string('cfg_path', './configs/arc_vgg19_2ed_stage.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def main(_argv):
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

    model = ArcFishStackModel(basemodel=basemodel,
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True, cfg=cfg)
    model.summary(line_length=80)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    File_log_name = 'logs/multistage_Ids10Test_tent_vote.log'
    scores_session1, scores_session2, scores_session3, scores_session4 = reportAccu(model,cfg=cfg)
    printstr = f"{scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n"

    with open(File_log_name, encoding="utf-8", mode="a") as data:
        data.write(printstr)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
