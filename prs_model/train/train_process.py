#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: train_process.py
    Description:
    
Created by YongBai on 2020/5/7 3:49 PM.
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from ..preprocess.dataloader import load_data_fsubset
import logging
import pickle


def cv_train(estimator, trian_save_fname, model_save_fname, n_fold=10, n_file=20):
    # file index from 21 to 24 as independent test dataset
    np.random.seed(123)
    file_arr = np.random.permutation(list(range(1, n_file + 1)))
    kf = KFold(n_splits=n_fold)

    for k_i, (train_index, val_index) in enumerate(kf.split(file_arr)):

        logging.info('{}/{}-th fold CV'.format(k_i+1, n_fold))

        train_file_index = file_arr[train_index]
        val_file_index = file_arr[val_index]

        trian_save_fname_split = os.path.splitext(trian_save_fname)
        train_save_finalname = '{}_cv{}_{}'.format(trian_save_fname_split[0],
                                                   k_i, trian_save_fname_split[1])

        model_save_fname_split = os.path.splitext(model_save_fname)
        model_save_finalname = '{}_cv{}_{}'.format(model_save_fname_split[0],
                                                   k_i, model_save_fname_split[1])

        _core(estimator, train_file_index, val_file_index,
              trian_save_fname=train_save_finalname, model_save_fname=model_save_finalname)

        if estimator.verbose > 0:
            print(' {}/{}-th fold CV finished, Epoch: {}'.format(k_i+1, n_fold, estimator.curr_epoch))


def _core(estimator, train_file_index, val_file_index, trian_save_fname=None, model_save_fname=None):

    if trian_save_fname is not None and os.path.exists(trian_save_fname):
        os.remove(trian_save_fname)
    if model_save_fname is not None and os.path.exists(model_save_fname):
        os.remove(model_save_fname)

    # write train result header
    if trian_save_fname is not None:
        with open(trian_save_fname, 'a') as tr_f:
            tr_f.write('Epoch,{}\n'.format(estimator.monitor))

    for epoch in range(estimator.epoches):
        if estimator.verbose > 0:
            print('\nEpoch: {}/{}'.format(epoch + 1, estimator.epoches))

        i_n = 1
        data_chunks = len(train_file_index)
        for i_file_name, train_x, train_y in load_data_fsubset(
                fsubset_folder_name=estimator.feature_index, file_ind_arr=train_file_index):

            estimator.fit(train_x, train_y)

            if estimator.verbose > 0:
                train_str = '{}/{}: [{}],{}'.format(i_n, data_chunks, i_file_name, train_x.shape)
                loss = mean_squared_error(train_y, estimator.predict(train_x))
                train_str = train_str + ' loss(mse): {}'.format(loss)
                print(train_str)

            i_n += 1

        pred_y = []
        true_y = []
        for _, val_x, val_y in load_data_fsubset(fsubset_folder_name=estimator.feature_index,
                                                 file_ind_arr=val_file_index):
            pred_y.extend(estimator.predict(val_x))
            true_y.extend(val_y)

        val_metirc = estimator.monitor_func(true_y, pred_y)
        if estimator.verbose > 0:
            print('Validation on {}, {}: {}'.format(val_file_index, estimator.monitor, val_metirc))

        # save the train result
        if trian_save_fname is not None:
            with open(trian_save_fname, 'a') as tr_f:
                tr_f.write('{},{.7f}\n'.format(epoch, val_metirc))

        _check = np.where(estimator.mode in ['max', 'auto'],
                          estimator.best_metric < val_metirc + estimator.tol,
                          estimator.best_metric > val_metirc - estimator.tol)
        if _check:
            if estimator.verbose > 0:
                print('Performance improved, best metric: {} -> {}'.format(
                    estimator.best_metric, val_metirc))
            estimator.best_metric = val_metirc
            estimator.curr_not_changed = 0

            if model_save_fname is not None and os.path.exists(model_save_fname):
                print('Model saved at {}'.format(model_save_fname))
                os.remove(model_save_fname)
                pickle.dump(estimator, open(model_save_fname, 'wb'))
        else:
            estimator.curr_not_changed += 1
            if estimator.verbose > 0:
                print('performance not improved, best metric {}'.format(estimator.best_metric))

        estimator.curr_epoch = epoch + 1
        if estimator.curr_not_changed > estimator.patience:
            break

















