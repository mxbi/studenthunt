import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
import pickle
from sklearn.preprocessing import PolynomialFeatures

import config
conf = config.get_config()

runtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
print('Run:', runtime)

model_name = 'model2_polyxgb'

def create_feature_map(features):
    outfile = open(conf['CACHE_LOC'] + 'xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap=conf['CACHE_LOC'] + 'xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

# Whether to add model to L2 stack
write_meta = True

print('Running ' + model_name + '.py!')

##############################
print('Loading data ...')
x_train, y_train, id_train, x_valid, y_valid, id_valid, x_test, id_test = pickle.load(open(conf['CACHE_LOC'] + 'data.bin', 'rb'))

poly = PolynomialFeatures(degree=2)
x_train = poly.fit_transform(x_train.fillna(0))
x_valid = poly.transform(x_valid.fillna(0))
x_test = poly.transform(x_test.fillna(0))
# x_train = np.hstack([x_train, x_train**2, x_train**3, x_train**4, x_train**5])
# x_valid = np.hstack([x_valid, x_valid**2, x_valid**3, x_valid**4, x_valid**5])

print('Column count:', x_train.shape[1])

##############################
print('Running model for valid')

# Set parameters
params = {}
params['booster'] = 'gblinear'
params['eta'] = 0.3  # Learning rate
params['eval_metric'] = 'rmse'
params['max_depth'] = 3
params['alpha'] = 0
params['colsample_bytree'] = 0.9
params['subsample'] = 0.9
params['silent'] = 1

# Convert data to xgboost format
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(x_test)

# List of datasets to evaluate on, last one is used for early stopping
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Train!
clf = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=500, verbose_eval=25)

# Save model
clf.save_model(conf['MODEL_LOC'] + model_name + '_valid_model_' + runtime + '.txt')

# Get best number of trees for booster
ntrees = 500

# Predict
p_train = clf.predict(d_train)
p_valid = clf.predict(d_valid)
p_test_mini = clf.predict(d_test)

# Write submission
sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test_mini
sub.to_csv(conf['SUB_LOC'] + model_name + '_mini_' + runtime + '.csv', index=False)

##############################
print('Running model for test')

d_train_full = xgb.DMatrix(np.vstack([x_train, x_valid]), label=y_train.tolist() + y_valid.tolist())

# List of datasets to evaluate on, last one is used for early stopping
watchlist = [(d_train_full, 'train')]

# Train!
clf = xgb.train(params, d_train_full, ntrees, watchlist, verbose_eval=25)

# Predict
p_train_full = clf.predict(d_train_full)
p_test = clf.predict(d_test)

# Save model
clf.save_model(conf['MODEL_LOC'] + model_name + '_test_model_' + runtime + '.txt')

# Write params
f = open(conf['INFO_LOC'] + model_name + '_params_' + runtime + '.txt', 'w')
f.write(str(params))
f.close()

# Write L2 indices
if write_meta:
    sub = pd.DataFrame({'id': id_train, model_name: p_train})
    sub.to_csv(conf['META_LOC'] + 'train/' + model_name + '.csv', index=False)

    sub = pd.DataFrame({'id': id_valid, model_name: p_valid})
    sub.to_csv(conf['META_LOC'] + 'valid/' + model_name + '.csv', index=False)

    sub = pd.DataFrame({'id': id_test, model_name: p_test_mini})
    sub.to_csv(conf['META_LOC'] + 'test_mini/' + model_name + '.csv', index=False)

    sub = pd.DataFrame({'id': id_train.tolist() + id_valid.tolist(), model_name: p_train_full})
    sub.to_csv(conf['META_LOC'] + 'train_full/' + model_name + '.csv', index=False)

    sub = pd.DataFrame({'id': id_test, model_name: p_test})
    sub.to_csv(conf['META_LOC'] + 'test/' + model_name + '.csv', index=False)

# Write submission
sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test
sub.to_csv(conf['SUB_LOC'] + model_name + '_' + runtime + '.csv', index=False)
