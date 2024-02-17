import numpy as np
import pandas as pd
import sklearn
from sklearn import *
import lightgbm as lgbm

np.random.seed(0)

df = sklearn.datasets.make_regression(10000)

X, y = df[0], df[1]
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=.25)

X_train_lgbm = lgbm.Dataset(
    data=X_train,
    label=y_train,
    reference=None,
    weight=None,
    group=None,
    init_score=None,
    feature_name='auto',
    categorical_feature='auto',
    params=None,
    free_raw_data=False,
)

X_valid_lgbm = lgbm.Dataset(
    data=X_valid,
    label=y_valid,
    reference=None,
    weight=None,
    group=None,
    init_score=None,
    feature_name='auto',
    categorical_feature='auto',
    params=None,
    free_raw_data=False,
)

def mse_custom_train(preds, data):
    y_true = data.get_label()
    y_pred = preds
    residual = (y_true - y_pred).astype("float")

    grad = np.where(residual < 0, -1. * residual, -1. * residual)
    hess = np.where(residual < 0, 1. * 1., 1. * 1.)

    return grad, hess


def mse_custom_eval(preds, data):
    y_true = data.get_label()
    y_pred = preds
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) * 1., (residual ** 2) * 1.)

    return "mse_custom", np.mean(loss), False


params = {
    'objective': mse_custom_train,
    'metric': ["mse"],
    'learning_rate': .1,
}

model = lgbm.train(
    params=params,
    train_set=X_train_lgbm,
    num_boost_round=1000,
    valid_sets=[X_train_lgbm, X_valid_lgbm, ],
    valid_names=None,
    init_model=None,
    feval=mse_custom_eval,
    # feature_name='auto',
    # categorical_feature='auto',
    keep_training_booster=False,
    callbacks=None,
)
