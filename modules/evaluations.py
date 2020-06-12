"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import os
import cv2
import bcolz
import numpy as np
import tqdm
from sklearn.model_selection import KFold
import tensorflow as tf
from .utils import l2_norm
# sklearn scipy used for EER cal.
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import json
from modules.dataset import loadTestDS
from modules.LoadFishDataUtil import LoadFishDataUtil


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]

    return ccropped_imgs


def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10, cfg=None):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    if cfg['head_type'] == 'IoMHead':
        # dist = dist/(cfg['q']*cfg['embd_shape']) # should divide by the largest distance
        dist = dist / (tf.math.reduce_max(dist).numpy() + 10)  # should divide by the largest distance
    print("[*] dist {}".format(dist))
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    auc = metrics.auc(fpr, tpr)
    # print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)

    return tpr, fpr, accuracy, best_thresholds, auc, eer


'''


def reportAccu(model_2ed,cfg=None):
    # load lable dict
    with open('data/Label_dict_2ed.json', 'r') as fp:
        Label_dict_2ed = json.load(fp)

    test_data_dir = './data/tmp_tent/test/SESSION1_LT'
    scores_session1 =getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION2'
    scores_session2 =getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION3'
    scores_session3 = getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION4'
    scores_session4 = getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    return scores_session1,scores_session2,scores_session3,scores_session4



def getAccByvote(test_data_dir,cfg=None,LableDict=None,model=None):
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE=32, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess1_test_dataset, sess1_class_num = testloadData.loadTestFishData()

    dataset = loadTestDS(test_data_dir,BATCH_SIZE=64,cfg=cfg,LableDict=LableDict)
    sess1_class_num = 10
    ds_it = iter(sess1_test_dataset)

    result = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}

    num_batch = 0
    for batch in dataset:
        thisbatch = next(ds_it)
        imgs, label = thisbatch
        output = model.predict(thisbatch)
        output = tf.argmax(tf.transpose(output))
        print(output)
        print(label)
        # for i in range(output.shape[0]):
        #     mylabel = label[i].numpy()[0][0]
        #     result[mylabel].append(int(output[i]))

    # print(result)
    final = {}
    correct = 0
    for i in range(sess1_class_num):
        lst = result[i]
        modeval = [x for x in set(lst) if lst.count(x) > 1]
        modeval = modeval[0]
        final[i] = modeval
        if i == modeval:
            correct = correct + 1
    return correct/sess1_class_num

'''


def reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES, model_2ed):
    test_data_dir = './data/tmp_tent/test/SESSION1_LT'
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess1_test_dataset, sess1_class_num = testloadData.loadTestFishData()

    scores_session1 = getAccByvote(model_2ed, test_data_dir, sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT,
                                   CLASS_NAMES)

    test_data_dir = './data/tmp_tent/test/SESSION2'
    scores_session2 = getAccByvote(model_2ed, test_data_dir, sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT,
                                   CLASS_NAMES)

    test_data_dir = './data/tmp_tent/test/SESSION3'
    scores_session3 = getAccByvote(model_2ed, test_data_dir, sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT,
                                   CLASS_NAMES)

    test_data_dir = './data/tmp_tent/test/SESSION4'
    scores_session4 = getAccByvote(model_2ed, test_data_dir, sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT,
                                   CLASS_NAMES)

    return scores_session1, scores_session2, scores_session3, scores_session4



def reportAccu_ds(cfg, model_2ed):
    test_data_dir = './data/stage2/SESSION1_LT/'

    scores_session1 = getAccByvote_ds(model_2ed, test_data_dir, cfg)

    test_data_dir = './data/stage2/SESSION2/'
    scores_session2 = getAccByvote_ds(model_2ed, test_data_dir, cfg)

    test_data_dir = './data/stage2/SESSION3/'
    scores_session3 = getAccByvote_ds(model_2ed, test_data_dir, cfg)

    test_data_dir = './data/stage2/SESSION4/'
    scores_session4 = getAccByvote_ds(model_2ed, test_data_dir, cfg)

    return scores_session1, scores_session2, scores_session3, scores_session4



def getAccByvote_ds(model_2ed, test_data_dir, cfg, sess1_class_num=10):
    sess1_test_dataset = loadTestDS(test_data_dir, BATCH_SIZE=cfg['batch_size'], cfg=cfg)
    ds_it = iter(sess1_test_dataset)

    result = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}

    num_batch = 0
    for batch in sess1_test_dataset:
        imgs, label = next(ds_it)
        output = model_2ed.predict(imgs)
        output = tf.argmax(tf.transpose(output))
        print('[***]output',output)
        print('[***]label',label)
        for i in range(output.shape[0]):
            mylabel = label[i].numpy()
            result[mylabel].append(int(output[i]))

    print(result)
    final = {}
    correct = 0
    for i in result.keys():
        lst = result[i]
        modeval = [x for x in set(lst) if lst.count(x) > 1]
        if (len(modeval)>0):
            modeval = modeval[0]
            final[i] = modeval
        else:
            modeval = -9
            final[i] = modeval

        if i == modeval:
            correct = correct + 1
    return correct / sess1_class_num


def getAccByvote(model_2ed, test_data_dir, sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES):
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess1_test_dataset, sess1_class_num = testloadData.loadTestFishDataWithname()
    ds_it = iter(sess1_test_dataset)

    result = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}

    num_batch = 0
    for batch in sess1_test_dataset:
        imgs, label = next(ds_it)
        output = model_2ed.predict(imgs)
        output = tf.argmax(tf.transpose(output))
        for i in range(output.shape[0]):
            mylabel = label[i].numpy()[0][0]
            result[mylabel].append(int(output[i]))

    # print(result)
    final = {}
    correct = 0
    for i in range(sess1_class_num):
        lst = result[i]
        modeval = [x for x in set(lst) if lst.count(x) > 1]
        if (len(modeval)>0):
            modeval = modeval[0]
            final[i] = modeval
        else:
            modeval = -9
            final[i] = modeval

        if i == modeval:
            correct = correct + 1
    return correct / sess1_class_num
