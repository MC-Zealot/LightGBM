# coding: utf-8
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import numpy as np
from scipy import special, optimize


# define cost and eval functions
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
    hess = np.where(residual < 0, 2, 2 * 1.15)
    return grad, hess

def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) , (residual ** 2) * 1.15)
    return "custom_asymmetric_eval", np.mean(loss), False


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
# lgb_train = lgb.Dataset(X_train, y_train, init_score=np.full_like(y_train, logloss_init_score(y_train), dtype=float))
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, init_score=np.full_like(y_test, logloss_init_score(y_test), dtype=float))

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': custom_asymmetric_train,
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
                num_boost_round=35,
                valid_sets=lgb_eval,
                feval=custom_asymmetric_valid,
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
