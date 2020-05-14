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


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    """get validation data"""
    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    agedb_30, agedb_30_issame = get_val_pair(data_path,
                                             'agedb_align_112/AgeDB/agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame


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
                  nrof_folds=10,cfg=None):
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
        dist = dist / (tf.math.reduce_max(dist).numpy()+10)  # should divide by the largest distance
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

    return tpr, fpr, accuracy, best_thresholds,auc,eer


def calculate_roc_cosine(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10,cfg=None):
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

    embeddings1_norm = tf.reduce_sum(embeddings1, axis=1)
    embeddings2_norm = tf.reduce_sum(embeddings2, axis=1)
    factor = embeddings1_norm + embeddings2_norm

    diff = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=1)

    dist = tf.divide(diff, factor)
    # a_n_vec = tf.divide(a_n_product, denom2)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # if cfg['head_type'] == 'IoMHead':
    #     # dist = dist/(cfg['q']*cfg['embd_shape']) # should divide by the largest distance
    #     dist = dist / (cfg['q']*100 )  # should divide by the largest distance
    # print(dist)
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
    return tpr, fpr, accuracy, best_thresholds

def evaluate(embeddings, actual_issame, nrof_folds=10,cfg=None):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]# 隔行采样
    embeddings2 = embeddings[1::2]# 隔行采样
    tpr, fpr, accuracy, best_thresholds,auc,eer= calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds, cfg=cfg)

    return tpr, fpr, accuracy, best_thresholds,auc,eer


def perform_val(embedding_size, batch_size, model,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=False,cfg=None):
    """perform val"""

    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        if is_ccrop:
            batch = ccrop_batch(batch)

        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            batch = ccrop_batch(batch)
            emb_batch = model(batch)
        # print(emb_batch)

        embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        # print(embeddings)
    tpr, fpr, accuracy, best_thresholds,auc,eer = evaluate(
        embeddings, issame, nrof_folds,cfg)

    return accuracy.mean(), best_thresholds.mean(),auc,eer,embeddings



def reportAccu(model_2ed,cfg=None):
    # load lable dict
    with open('data/Label_dict_2ed.json', 'r') as fp:
        Label_dict_2ed = json.load(fp)

    test_data_dir = './tmp_tent/test/SESSION1_LT'
    scores_session1 =getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION2'
    scores_session2 =getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION3'
    scores_session3 = getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    test_data_dir = './data/tmp_tent/test/SESSION4'
    scores_session4 = getAccByvote(test_data_dir,cfg=cfg,LableDict=Label_dict_2ed,model=model_2ed)

    return scores_session1,scores_session2,scores_session3,scores_session4



def getAccByvote(test_data_dir,cfg=None,LableDict=None,model=None):
    dataset = loadTestDS(test_data_dir,BATCH_SIZE=64,cfg=cfg,LableDict=LableDict)
    sess1_class_num = 10
    ds_it = iter(dataset)

    result = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}

    num_batch = 0
    for batch in dataset:
        imgs, label = next(ds_it)
        output = model.predict(imgs)
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
        modeval = modeval[0]
        final[i] = modeval
        if i == modeval:
            correct = correct + 1
    return correct/sess1_class_num
