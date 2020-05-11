#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: lasso_estimator.py
    Description:
    
Created by YongBai on 2020/5/7 3:07 PM.
"""
from .base_estimator import LocalBase
from sklearn.linear_model import SGDRegressor


class LassoEstimator(LocalBase):
    def __init__(self, feature_index, alpha=0.000001,
                 eta0=0.00001, n_epoches=100, tol=0.000001,
                 patience=10, verbose=1, mode='min', monitor='mse'):
        # https://stackoverflow.com/questions/31443840/sgdregressor-nonsensical-result
        # set eta0 to small value
        # eta0: 0.0001-0.001, small: mse too high, and learning slow; large: mse low, learning quick
        # eta0: optimal=0.0002
        self.estimator = SGDRegressor(loss='huber', penalty='l1', alpha=alpha, eta0=eta0)
        super().__init__(feature_index, n_epoches=n_epoches,
                         patience=patience, tol=tol, verbose=verbose,
                         mode=mode, monitor=monitor)

    def fit(self, X, y):
        self.estimator.partial_fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)
