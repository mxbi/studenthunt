import pandas as pd
import numpy as np
import xgboost as xgb
from operator import itemgetter
import datetime
from sklearn.metrics import mean_squared_error
from keras.layers.advanced_activations import PReLU

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

print('Reading input data ...')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

########## FEATURE ENGINEERING ##########

for df in [df_train, df_test]:
    df['year'] = df.Date.apply(lambda x: int(x.split('-')[2]))
    df['month'] = df['Date'].apply(lambda x: int(x.split('-')[1]))
    df['date_dm'] = df['Date'].apply(lambda x: '-'.join(x.split('-')[:2]))
    df['Direction_Of_Wind'] = df['Direction_Of_Wind'].apply(lambda x: x + np.random.uniform(-10, 10))
    # df['pressure_range'] = df['Max_Atmospheric_Pressure'] - df['Min_Atmospheric_Pressure']
    # df['pollution_range'] = df['Max_Ambient_Pollution'] - df['Min_Ambient_Pollution']
    # df['breeze_range'] = df['Max_Breeze_Speed'] - df['Min_Breeze_Speed']

for i in df_train['Park_ID'].unique().tolist():
    df_train['Park' + str(i)] = (df_train['Park_ID'].values == i)
    df_test['Park' + str(i)] = (df_test['Park_ID'] == i)

date_encode = df_train.groupby('date_dm')['Footfall'].mean()
df_train['date_encode'] = df_train['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)
df_test['date_encode'] = df_test['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)

print(df_train)
print(df_train.groupby('Direction_Of_Wind')['Footfall'].mean())


drop_cols = ['Date', 'year', 'date_dm']
#########################################

print('Train size', df_train.shape, 'Test size', df_test.shape)

validation = True

if validation:
    y_train = df_train.loc[df_train.year <= 1998].loc[df_train.year > 1992]['Footfall']
    x_train = df_train.loc[df_train.year <= 1998].loc[df_train.year > 1992].drop(['ID', 'Footfall'] + drop_cols, 1)
else:
    print('VALIDATION OFF')
    y_train = df_train['Footfall']
    x_train = df_train.drop(['ID', 'Footfall'] + drop_cols, 1)

y_valid = df_train.loc[df_train.year > 1998]['Footfall']
x_valid = df_train.loc[df_train.year > 1998].drop(['ID', 'Footfall'] + drop_cols, 1)

id_test = df_test['ID']
x_test = df_test.drop(['ID'] + drop_cols, 1)

print('Train size', x_train.shape, 'Valid size', x_test.shape)
print('Columns:', x_train.columns, '\n')

# Set parameters
params = {}
params['booster'] = 'gbtree'
params['eta'] = 0.02 # Learning rate
params['eval_metric'] = 'rmse'
params['max_depth'] = 4
params['colsample_bytree'] = 0.9
params['subsample'] = 0.9
params['silent'] = 1

# Convert data to xgboost format
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

# List of datasets to evaluate on, last one is used for early stopping
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Train!
# Third value is number of rounds (n_estimators), early_stopping_rounds stops training when it hasn't improved for that number of rounds
clf = xgb.train(params, d_train, 20000, watchlist, early_stopping_rounds=500, verbose_eval=25)

# Predict
d_test = xgb.DMatrix(x_test)
p_test = clf.predict(d_test) # Returns array with *single column*, probability of 1
p_valid = clf.predict(d_valid)

print(get_importance(clf, list(x_train.columns.values)))

sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p_test
sub.to_csv('simple_xgb_submission.csv', index=False)

p2_test = []
for x in p_test:
    if 900 < x < 1600:
        p2_test.append(x)
    else:
        p2_test.append('')

sub = pd.DataFrame()
sub['ID'] = id_test
sub['Footfall'] = p2_test
sub.to_csv('super_submission.csv', index=False)

#### CONFUSION

# for i in [(700, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 1200), (1200, 1300), (1300, 1400), (1400, 1500), (1500, 1600), (1600, 1700), (1700, 1800)]:
#     ia, ib = i
#     x=pd.DataFrame()
#     x['y'] = y_valid
#     x['p'] = p_valid
#     xx = x.loc[x.y > ia].loc[x.y < ib]
#     print(len(xx))
#     print(np.sqrt(mean_squared_error(xx['y'], xx['p'])))

# #### LEAK
#
# leak_thresh = 100
# yl_valid = []
# for a, b in zip(y_valid, p_valid):
#     diff = abs(a - b)
#     if diff < leak_thresh:
#         yl_valid.append(1)
#     else:
#         yl_valid.append(0)
#
# dl_valid = xgb.DMatrix(x_valid, label=yl_valid)
#
# watchlist = [(d_valid, 'valid')]
#
# params = {}
# params['objective'] = 'binary:logistic'
# params['booster'] = 'gbtree'
# params['eta'] = 0.02 # Learning rate
# params['eval_metric'] = 'auc'
# params['max_depth'] = 6
# params['colsample_bytree'] = 0.9
# params['subsample'] = 0.9
# params['silent'] = 1
#
# clf = xgb.train(params, d_valid, 20000, watchlist, early_stopping_rounds=500, verbose_eval=25)
