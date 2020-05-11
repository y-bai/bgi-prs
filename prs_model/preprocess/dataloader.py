#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: dataloader.py
    Description:
    
Created by YongBai on 2020/4/16 5:47 PM.
"""

import os
import numpy as np
import pandas as pd
from prs_utils import get_config


def load_data_fsubset(fsubset_folder_name='1-2000', file_ind_arr=range(1, 25)):
    """
    Load data based on given subset of features

    Parameters
    ----------
    fsubset_folder_name: str, default='1-2000'
        the folder name of subset features

    file_ind_arr: list, np.array or range
        files index list. The Files were split according to row axis.

    Returns
    -------
        yield X and y
    """

    r_dir = get_config()['data']['input_data_fdir']
    fsubset_dir = os.path.join(r_dir, fsubset_folder_name)

    if not os.path.isdir(fsubset_dir):
        raise FileNotFoundError('Folder not existing: {}'.format(fsubset_dir))

    for i in file_ind_arr:
        x_f_name = '{}.data.txt'.format(i)
        x_file = os.path.join(fsubset_dir, x_f_name)
        y_file = os.path.join(r_dir, '{}.real.height.txt'.format(i))

        yield '{}/{}'.format(fsubset_folder_name, x_f_name), \
              pd.read_csv(x_file, dtype=np.float32, header=None).values, \
              np.loadtxt(y_file, dtype=np.float32)







