import numpy as np
import pandas as pd
import sklearn
from sklearn import *
import lightgbm as lgbm
from sklearn.metrics import auc, mean_squared_error, roc_curve

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

def huber_custom_train_v2(preds, data):

    y_true = data.get_label()
    y_pred = preds
    residual = (y_pred - y_true).astype("float")

    alpha = .9

    grad = np.where(np.abs(residual) <= alpha, residual, np.sign(residual) * alpha)
    hess = np.where(residual < 0, 1. * 1., 1. * 1.)

    return grad, hess

def mse_custom_eval(preds, data):
    y_true = data.get_label()
    y_pred = preds
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) * 1., (residual ** 2) * 1.)

    return "mse_custom", np.mean(loss), False


def huber_custom_eval_v2(preds, data):
    y_true = data.get_label()
    y_pred = preds
    residual = (y_pred - y_true).astype("float")
    alpha = .9
    loss = np.where(np.abs(residual) <= alpha, .5 * ((residual) ** 2), alpha * np.abs(residual) - .5 * (alpha ** 2))

    return "huber_custom", np.mean(loss), False


params = {
    'objective': huber_custom_train_v2,
    'metric': ["mse"],
    'learning_rate': .1,
    'early_stopping_rounds=100':100,
    'verbose_eval':100
}

gbm = lgbm.train(
    params=params,
    train_set=X_train_lgbm,
    num_boost_round=1000,
    valid_sets=[X_train_lgbm, X_valid_lgbm, ],
    valid_names=None,
    init_model=None,
    feval=huber_custom_eval_v2,
    keep_training_booster=False,
    callbacks=None,
)

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
# eval
mse = mean_squared_error(y_valid, y_pred)
print(f'The mse of prediction is: {mse}')

