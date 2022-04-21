from getFilePath import getFilePath
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import metrics
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None) # all cols
pd.set_option('display.width', 161)

# Read the data
filepath = getFilePath("ks-projects-201801.csv")
# read the data and store data in DataFrame titled melbourne_data
ks = pd.read_csv(filepath, parse_dates=['deadline', 'launched'])

#print(ks.head(10))

#print(pd.unique(ks.state))

#print(ks.groupby('state')['ID'].count())

# Drop live projects
ks = ks.query('state != "live"')
# Add outcome column, "successful" == 1, others are 0
ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

ks = ks.assign(hour=ks.launched.dt.hour,
day=ks.launched.dt.day,
month=ks.launched.dt.month,
year=ks.launched.dt.year)

#print(ks.head(10))

cat_features = ['category', 'currency', 'country']
encoder = LabelEncoder()
# Apply the label encoder to each column
encoded = ks[cat_features].apply(encoder.fit_transform)
#print(encoded.head(10))

# Since ks and encoded have the same index and I can easily join them
data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)
print(data.head(10))

# could use sklearn.model_selection.StratifiedShuffleSplit
valid_fraction = 0.1
valid_size = int(len(data) * valid_fraction)
train = data[:-2 * valid_size]
valid = data[-2 * valid_size:-valid_size]
test = data[-valid_size:]

for each in [train, valid, test]:
    print(f"Outcome fraction = {each.outcome.mean():.4f}")

# Train a LightGBM model

feature_cols = train.columns.drop('outcome')
dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                early_stopping_rounds=10, verbose_eval=False)

# predict & score
ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['outcome'], ypred)

print(f"Test AUC score: {score}")
# Test AUC score: 0.747615303004287
