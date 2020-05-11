#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: main.py.py
    Description:
    
Created by YongBai on 2020/4/16 5:38 PM.
"""
import logging
from prs_model import *
from prs_utils import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    logger.info('start training')

    r_dir = get_config()['data']['input_data_fdir']
    feature_index_str = '1-2000'
    estimator = LassoEstimator(feature_index_str)

    train_save_fname = os.path.join(r_dir, feature_index_str, 'train_result.csv')
    model_save_fname = os.path.join(r_dir, feature_index_str, 'train_model.pkl')
    cv_train(estimator, train_save_fname, model_save_fname)

    logger.info('end')
    print("DONE")