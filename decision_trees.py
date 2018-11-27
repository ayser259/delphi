import numpy as np
import pandas as pd
from sklearn import preprocessing,tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.neighbors import KNeighborsClassifier

from data_load import get_encoded_data,get_one_hot_encoded_data

###========================= TREE Full Data Label Encoded Data =================
df = get_encoded_data('data.csv')[0]
# encoded_dict_list = get_encoded_data('data.csv')[1]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

# Leave one out validation
X = np.array(x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

# Calculate accuracy
TREE_FD_LE_mean = np.array(accuracy).mean()
TREE_FD_LE_variance = np.array(accuracy).std() * 2

###==========================TREE Full Data One Hot Encoded  ===================

df = get_one_hot_encoded_data('data.csv')
x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

X = np.array(x_df) # convert dataframe into np array
y = np.array(y_df) # convert dataframe into np array

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

# Calculate accuracy
TREE_FD_OHE_mean = np.array(accuracy).mean()
TREE_FD_OHE_variance = np.array(accuracy).std() * 2

###===============TREE SFM Data Label Encoded Data  =======================
df = get_encoded_data('data.csv')[0]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

SFM = tree.DecisionTreeClassifier().fit(x_df,y_df)

modelNew = SelectFromModel(SFM, prefit = True)
fs_x_df = modelNew.transform(x_df)
feature_idx = modelNew.get_support()
feature_name = x_df.columns[feature_idx]
print("Features Remaining After LE,SFM,LR",list(feature_name))
X = np.array(fs_x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

TREE_SFM_LE_mean = np.array(accuracy).mean()
TREE_SFM_LE_variance = np.array(accuracy).std() * 2

###===============TREE SFM Data One Hot Encoded Data  =======================
df = get_one_hot_encoded_data('data.csv')

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

SFM = tree.DecisionTreeClassifier().fit(x_df,y_df)

modelNew = SelectFromModel(SFM, prefit = True)
fs_x_df = modelNew.transform(x_df)
feature_idx = modelNew.get_support()
feature_name = x_df.columns[feature_idx]
print("Features Remaining After OHE,SFM,LR",list(feature_name))
X = np.array(fs_x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

TREE_SFM_OHE_mean = np.array(accuracy).mean()
TREE_SFM_OHE_variance = np.array(accuracy).std() * 2

###===============TREE RFE Data Label Encoded Data  =======================
df = get_encoded_data('data.csv')[0]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

X = np.array(x_df)
y = np.array(y_df)

RFEM = tree.DecisionTreeClassifier()
rfe = RFE(RFEM, 5)
fit = rfe.fit(X, y)
featureidx = fit.get_support()
feature_names = list(x_df.columns[featureidx])
x_df = x_df[feature_names]

print("Features Remaining After LE,RFE,NB",list(feature_names))

X = np.array(x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

TREE_RFE_LE_mean = np.array(accuracy).mean()
TREE_RFE_LE_variance = np.array(accuracy).std() * 2

###===============TREE RFE Data One Hot Encoded Data  =======================
df = get_one_hot_encoded_data('data.csv')

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

X = np.array(x_df)
y = np.array(y_df)

RFEM =tree.DecisionTreeClassifier()
rfe = RFE(RFEM, 17)
fit = rfe.fit(X, y)
featureidx = fit.get_support()
feature_names = list(x_df.columns[featureidx])
x_df = x_df[feature_names]

print("Features Remaining After LE,RFE,LR",list(feature_names))

X = np.array(x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))

TREE_RFE_OHE_mean = np.array(accuracy).mean()
TREE_RFE_OHE_variance = np.array(accuracy).std() * 2

#Printing Results
print("[NB,Full Data,Label Encoded] Accuracy: %0.2f (+/- %0.2f)" % (TREE_FD_LE_mean, TREE_FD_LE_variance))
print("[NB,Full Data,One Hot Encoded] Accuracy: %0.2f (+/- %0.2f)" % (TREE_FD_OHE_mean, TREE_FD_OHE_variance))
print("[TREE,SFM,Label Encoded] Accuracy: %0.2f (+/- %0.2f)" % (TREE_SFM_LE_mean, TREE_SFM_LE_variance))
print("[TREE,SFM,One Hot Encoded] Accuracy: %0.2f (+/- %0.2f)" % (TREE_SFM_OHE_mean, TREE_SFM_OHE_variance))
print("[TREE,RFE,Label Encoded]Accuracy: %0.2f (+/- %0.2f)" % (TREE_RFE_LE_mean, TREE_RFE_LE_variance))
print("[TREE,RFE,One Hot Encoded] Accuracy: %0.2f (+/- %0.2f)" % (TREE_RFE_OHE_mean, TREE_RFE_OHE_variance))
