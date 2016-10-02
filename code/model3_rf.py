import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import config
conf = config.get_config()

runtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
print('Run:', runtime)

model_name = 'model3_rf'

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

# poly = PolynomialFeatures(degree=2)
# x_train = poly.fit_transform(x_train.fillna(0))
# x_valid = poly.transform(x_valid.fillna(0))
# x_test = poly.transform(x_test.fillna(0))
x_train = x_train.fillna(0)
x_valid = x_valid.fillna(0)
x_test = x_test.fillna(0)
# x_train = np.hstack([x_train, x_train**2, x_train**3, x_train**4, x_train**5])
# x_valid = np.hstack([x_valid, x_valid**2, x_valid**3, x_valid**4, x_valid**5])

print('Column count:', x_train.shape[1])

##############################
print('Running model for valid')

clf = RandomForestRegressor(n_estimators=250, max_depth=8, n_jobs=-1, random_state=4242, verbose=9)

clf.fit(x_train, y_train)

# Predict
p_train = clf.predict(x_train)
p_valid = clf.predict(x_valid)
p_test_mini = clf.predict(x_test)

print('Train score:', np.sqrt(mean_squared_error(y_train, p_train)), 'Valid score:', np.sqrt(mean_squared_error(y_valid, p_valid)))

# Write submission
sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test_mini
sub.to_csv(conf['SUB_LOC'] + model_name + '_mini_' + runtime + '.csv', index=False)

##############################
print('Running model for test')

x_train_full = np.vstack([x_train, x_valid])
y_train_full = y_train.tolist() + y_valid.tolist()

# Train!
clf.fit(x_train_full, y_train_full)

# Predict
p_train_full = clf.predict(x_train_full)
p_test = clf.predict(x_test)

print('Train score:', np.sqrt(mean_squared_error(y_train_full, p_train_full)))

# Save model
clf.save_model(conf['MODEL_LOC'] + model_name + '_test_model_' + runtime + '.txt')

# # Write params
# f = open(conf['INFO_LOC'] + model_name + '_params_' + runtime + '.txt', 'w')
# f.write(str(params))
# f.close()

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
