#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: base_estimator.py
    Description:
    
Created by YongBai on 2020/5/7 2:58 PM.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error


class LocalBase(ABC):

    def __init__(self, feature_index, n_epoches=100, patience=10, tol=0.0001,
                 verbose=0, mode='min', monitor='mse'):
        self._feature_index = feature_index
        self._n_epoches = n_epoches
        self._verbose = verbose
        self._patience = patience

        self._mode = mode
        if self._mode in ['max', 'auto']:
            self._best_metric = -np.inf
        else:
            self._best_metric = np.inf

        self._curr_not_changed = 0

        self._monitor = monitor
        if self._monitor == 'mse':
            self._f = mean_squared_error

        self._curr_epoch = 0

        self._tol = tol

    @property
    def feature_index(self):
        return self._feature_index

    @property
    def epoches(self):
        return self._n_epoches

    @property
    def verbose(self):
        return self._verbose

    @property
    def patience(self):
        return self._patience

    @property
    def mode(self):
        return self._mode

    @property
    def monitor(self):
        return self._monitor

    @property
    def monitor_func(self):
        return self._f

    @property
    def tol(self):
        return self._tol

    @property
    def best_metric(self):
        return self._best_metric

    @best_metric.setter
    def best_metric(self, value):
        self._best_metric = value

    @property
    def curr_not_changed(self):
        return self._curr_not_changed

    @curr_not_changed.setter
    def curr_not_changed(self, value):
        self._curr_not_changed = value

    @property
    def curr_epoch(self):
        return self._curr_epoch

    @curr_epoch.setter
    def curr_epoch(self, value):
        self._curr_epoch = value

    @abstractmethod
    def fit(self, X, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        pass
