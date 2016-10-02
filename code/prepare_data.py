import pandas as pd
import numpy as np
import datetime
import pickle

import config
conf = config.get_config()

print('Running prepare_data.py!')

##############################
print('Reading input data ...')
df_train = pd.read_csv(conf['INPUT_LOC'] + 'train.csv')
df_test = pd.read_csv(conf['INPUT_LOC'] + '/test.csv')

##############################
print('Doing first feature engineering ...')

for df in [df_train, df_test]:
    df['year'] = df.Date.apply(lambda x: int(x.split('-')[2]))
    df['month'] = df['Date'].apply(lambda x: int(x.split('-')[1]))
    df['dateidix'] = df[['Date', 'Park_ID']].apply(lambda x: x[0] + str(x[1]), axis=1)
    df['date_dm'] = df['Date'].apply(lambda x: '-'.join(x.split('-')[:2]))
    df['Direction_Of_Wind'] = df['Direction_Of_Wind'].apply(lambda x: x + np.random.uniform(-10, 10))

for i in df_train['Park_ID'].unique().tolist():
    df_train['Park' + str(i)] = (df_train['Park_ID'].values == i)
    df_test['Park' + str(i)] = (df_test['Park_ID'] == i)

date_encode = df_train.groupby('date_dm')['Footfall'].mean()
df_train['date_encode'] = df_train['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)
df_test['date_encode'] = df_test['date_dm'].apply(lambda x: date_encode[x] + np.random.uniform(0, 25) if x in date_encode else 0)

##############################
print('Calculating lag/lead ...')

table = pd.concat([df_train, df_test]).set_index('dateidix')
vals = {}

for c in ['Direction_Of_Wind', 'Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
    print('Processing column', c)
    vals[c + '_lag1'] = []
    vals[c + '_lag2'] = []
    #vals[c + '_lead1'] = []
    #vals[c + '_lead2'] = []

# for c in ['Direction_Of_Wind', 'Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
#     print('Processing column', c)
#     vals[c + '_lag1'] = []
#     vals[c + '_lag2'] = []
#     vals[c + '_lead1'] = []
#     vals[c + '_lead2'] = []
#     for row in table[['Date', 'Park_ID']].values:
#         day, month, year = [int(x) for x in row[0].split('-')]
#         lag1_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 1, month, year]]) + str(row[1])
#         lag2_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 2, month, year]]) + str(row[1])
#         lead1_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 1, month, year]]) + str(row[1])
#         lead2_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 2, month, year]]) + str(row[1])
#
#         row_vals = []
#         for ix in [lag1_ix, lag2_ix, lead1_ix, lead2_ix]:
#             try:
#                 val = table.ix[lag1_ix][c]
#             except KeyError:
#                 val = np.nan
#             row_vals.append(val)
#
#         vals[c + '_lag1'].append(row_vals[0])
#         vals[c + '_lag2'].append(row_vals[1])
#         vals[c + '_lead1'].append(row_vals[2])
#         vals[c + '_lead2'].append(row_vals[3])

for i, row in enumerate(table[['Date', 'Park_ID']].values):
    if i % 10000 == 0:
        print(i)
    day, month, year = [int(x) for x in row[0].split('-')]
    lag1_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 1, month, year]]) + str(row[1])
    lag2_ix = '-'.join([str(x).rjust(2, '0') for x in [day - 2, month, year]]) + str(row[1])
    #lead1_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 1, month, year]]) + str(row[1])
    #lead2_ix = '-'.join([str(x).rjust(2, '0') for x in [day + 2, month, year]]) + str(row[1])

    for c in ['Direction_Of_Wind', 'Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
        row_vals = []
        for ix in [lag1_ix, lag2_ix]:#, lead1_ix, lead2_ix]:
            try:
                val = table.ix[lag1_ix][c]
            except KeyError:
                val = np.nan
            row_vals.append(val)

        vals[c + '_lag1'].append(row_vals[0])
        vals[c + '_lag2'].append(row_vals[1])
        #vals[c + '_lead1'].append(row_vals[2])
        #vals[c + '_lead2'].append(row_vals[3])

# for c in ['Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
#     #print('Processing column', c)
#     vals[c + '_weekavg'] = []
#
# for i, row in enumerate(table[['Date', 'Park_ID']].values):
#     if i % 10000 == 0:
#         print(i)
#     ixs = []
#     day, month, year = [int(x) for x in row[0].split('-')]
#     for ix in [0, -1, -2, -3, -4, -5, -6]:
#         ixs.append('-'.join([str(x).rjust(2, '0') for x in [day + ix, month, year]]) + str(row[1]))
#
#     for c in ['Average_Breeze_Speed', 'Var1', 'Average_Atmospheric_Pressure', 'Average_Moisture_In_Park']:
#         ls = []
#         for ix in ixs:
#             try:
#                 val = table.ix[ix][c]
#             except KeyError:
#                 val = np.nan
#             ls.append(val)
#
#         vals[c + '_weekavg'].append(np.nanmean(ls))


for a, b in vals.items():
    df_train[a] = b[:len(df_train)]
    df_test[a] = b[len(df_train):]

for c in ['Var1', 'Average_Atmospheric_Pressure', 'Max_Atmospheric_Pressure', 'Min_Atmospheric_Pressure', 'Min_Ambient_Pollution', 'Max_Ambient_Pollution']:
    print('Imputing', c)
    train_means = df_train.groupby('Date')[c].mean()
    test_means = df_test.groupby('Date')[c].mean()
    df_train[c + '_means'] = df_train['Date'].apply(lambda x: train_means[x] if x in train_means else np.nan)
    df_test[c + '_means'] = df_test['Date'].apply(lambda x: test_means[x] if x in test_means else np.nan)

##############################
print('Splitting data ...')

drop_cols = ['Date', 'year', 'date_dm', 'dateidix']

print('Train size', df_train.shape, 'Test size', df_test.shape)

id_train = df_train.loc[df_train.year <= 1998]['ID']
y_train = df_train.loc[df_train.year <= 1998]['Footfall']
x_train = df_train.loc[df_train.year <= 1998].drop(['ID', 'Footfall'] + drop_cols, 1)

id_valid = df_train.loc[df_train.year > 1998]['ID']
y_valid = df_train.loc[df_train.year > 1998]['Footfall']
x_valid = df_train.loc[df_train.year > 1998].drop(['ID', 'Footfall'] + drop_cols, 1)

id_test = df_test['ID']
x_test = df_test.drop(['ID'] + drop_cols, 1)

print('Train size', x_train.shape, 'Valid size', x_test.shape)
print('Columns:', x_train.columns.tolist(), '\n')

##############################
print('Saving data ...')

pickle.dump([x_train, y_train, id_train, x_valid, y_valid, id_valid, x_test, id_test], open(conf['CACHE_LOC'] + 'data.bin', 'wb'), protocol=4)
