import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
import pickle

import config
conf = config.get_config()

runtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
print('Run:', runtime)

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
write_meta = False

print('Running model1_xgb.py!')

##############################
print('Loading data ...')
x_train, y_train, id_train, x_valid, y_valid, id_valid, x_test, id_test = pickle.load(open(conf['CACHE_LOC'] + 'data.bin', 'rb'))

for df in [x_train, x_valid, x_test]:
    df['mois_change_1day'] = df['Average_Moisture_In_Park_lag1'] - df['Average_Moisture_In_Park']
    df['breeze_change_1day'] = df['Average_Breeze_Speed_lag1'] - df['Average_Breeze_Speed']
    df['pressure_change_1day'] = df['Average_Atmospheric_Pressure_lag1'] - df['Average_Atmospheric_Pressure']
    df['dow_change_1day'] = df['Direction_Of_Wind_lag1'] - df['Direction_Of_Wind']
    #df.drop('Direction_Of_Wind', 1, inplace=True)

print('Column count:', x_train.shape[1])

##############################
print('Running model for valid')

# Set parameters
params = {}
params['booster'] = 'gbtree'
params['eta'] = 0.02  # Learning rate
params['eval_metric'] = 'rmse'
params['max_depth'] = 3
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
clf = xgb.train(params, d_train, 20000, watchlist, early_stopping_rounds=500, verbose_eval=25)

# Save model
clf.save_model(conf['MODEL_LOC'] + 'model1_xgb_valid_model_' + runtime + '.txt')

# Get best number of trees for booster
ntrees = clf.best_iteration + 1

# Predict
p_train = clf.predict(d_train, ntree_limit=ntrees)
p_valid = clf.predict(d_valid, ntree_limit=ntrees)
p_test_mini = clf.predict(d_test, ntree_limit=ntrees)

# Write submission
sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test_mini
sub.to_csv(conf['SUB_LOC'] + 'model1_xgb_mini_' + runtime + '.csv', index=False)

##############################
print('Running model for test')

d_train_full = xgb.DMatrix(pd.concat([x_train, x_valid]), label=y_train.tolist() + y_valid.tolist())

# List of datasets to evaluate on, last one is used for early stopping
watchlist = [(d_train_full, 'train')]

# Train!
clf = xgb.train(params, d_train_full, ntrees, watchlist, verbose_eval=25)

# Predict
p_train_full = clf.predict(d_train_full)
p_test = clf.predict(d_test)

# Save model
clf.save_model(conf['MODEL_LOC'] + 'model1_xgb_test_model_' + runtime + '.txt')

# Write feature importance
imp = get_importance(clf, x_train.columns.tolist())
print('Importance array:', imp)
f = open(conf['INFO_LOC'] + 'model1_xgb_importance_' + runtime + '.txt', 'w')
f.write(str(imp))
f.close()

# Write params
f = open(conf['INFO_LOC'] + 'model1_xgb_params_' + runtime + '.txt', 'w')
f.write(str(params))
f.close()

# Write L2 indices
if write_meta:
    sub = pd.DataFrame({'id': id_train, 'model1_xgb': p_train})
    sub.to_csv(conf['META_LOC'] + 'train/model1_xgb.csv', index=False)

    sub = pd.DataFrame({'id': id_valid, 'model1_xgb': p_valid})
    sub.to_csv(conf['META_LOC'] + 'valid/model1_xgb.csv', index=False)

    sub = pd.DataFrame({'id': id_test, 'model1_xgb': p_test_mini})
    sub.to_csv(conf['META_LOC'] + 'test_mini/model1_xgb.csv', index=False)

    sub = pd.DataFrame({'id': id_train.tolist() + id_valid.tolist(), 'model1_xgb': p_train_full})
    sub.to_csv(conf['META_LOC'] + 'train_full/model1_xgb.csv', index=False)

    sub = pd.DataFrame({'id': id_test, 'model1_xgb': p_test})
    sub.to_csv(conf['META_LOC'] + 'test/model1_xgb.csv', index=False)

# Write submission
sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test
sub.to_csv(conf['SUB_LOC'] + 'model1_xgb_' + runtime + '.csv', index=False)
