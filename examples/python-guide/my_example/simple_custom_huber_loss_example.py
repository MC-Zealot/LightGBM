# coding: utf-8
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import numpy as np
from scipy import special, optimize


def logloss_init_score(y):
    p = y.mean()
    p = np.clip(p, 1e-15, 1 - 1e-15)  # never hurts
    log_odds = np.log(p / (1 - p))
    return log_odds

def logloss_init_score_v2(y_true):
    # 样本初始值寻找过程
    res = optimize.minimize_scalar(
        lambda p: (y_true, p).sum(),
        bounds=(0, 1),
        method='bounded'
    )
    p = res.x
    log_odds = np.log(p / (1 - p))
    return log_odds

def huber_custom_train_v2(preds, data):

    y_true = data.get_label()
    y_pred = preds
    residual = (y_pred - y_true).astype("float")

    alpha = .9

    grad = np.where(np.abs(residual) <= alpha, residual, np.sign(residual) * alpha)
    hess = np.where(residual < 0, 1. * 1., 1. * 1.)

    return grad, hess

def gradient_hessian(preds, data):
    alpha = .9
    y_true = data.get_label()
    residual = (preds - y_true).astype('float')
    gradient = np.where(np.abs(residual) <= alpha, residual, np.sign(residual) * alpha)
    hessian = np.ones(preds.shape)

    return gradient, hessian

def huber_custom_eval_v2(preds, data):
    y_true = data.get_label()
    y_pred = preds
    residual = (y_pred - y_true).astype("float")
    alpha = 0.9
    loss = np.where(np.abs(residual) <= alpha, .5 * ((residual) ** 2), alpha * np.abs(residual) - .5 * (alpha ** 2))

    return "huber_custom", np.mean(loss), False

print('Loading data...')
# load or create your dataset
regression_example_dir = Path(__file__).absolute().parents[1] / '../regression'
df_train = pd.read_csv(str(regression_example_dir / 'regression.train'), header=None, sep='\t')
df_test = pd.read_csv(str(regression_example_dir / 'regression.test'), header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, init_score=np.full_like(y_train, logloss_init_score_v2(y_train), dtype=float))
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, init_score=np.full_like(y_test, logloss_init_score_v2(y_test), dtype=float))

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': huber_custom_train_v2,
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                feval=huber_custom_eval_v2,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
mse_test = mean_squared_error(y_test, y_pred)
print(f'The RMSE of prediction is: {rmse_test}')
print(f'The MSE of prediction is: {mse_test}')
