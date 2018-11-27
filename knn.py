import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier

from data_load import get_encoded_data,get_one_hot_encoded_data

###=============================== KNN Full Data Label Encoded  ============

df = get_encoded_data('data.csv')[0]
# encoded_dict_list = get_encoded_data('data.csv')[1]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

X = np.array(x_df)
y = np.array(y_df)

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) #
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

  KNN = KNeighborsClassifier(n_neighbors=3)
  model = KNN.fit(X_train, y_train)
  accuracy.append(KNN.score(X_test, y_test))

KNN_FD_LE_mean = np.array(accuracy).mean()
KNN_FD_LE_variance = np.array(accuracy).std() * 2

###============================== KNN Full Data One Hot Encoded  ==============
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

  KNN = KNeighborsClassifier(n_neighbors=3)
  model = KNN.fit(X_train, y_train)
  accuracy.append(KNN.score(X_test, y_test))

# Calculate accuracy
KNN_FD_OH_mean = np.array(accuracy).mean()
KNN_FD_OH_variance = np.array(accuracy).std() * 2

#Printing Results
print("[KNN,Full Data,Label Encoded] Accuracy: %0.2f (+/- %0.2f)" % (KNN_FD_LE_mean, KNN_FD_LE_variance))
print("[KNN,Full Data,One Hot Encoded] Accuracy: %0.2f (+/- %0.2f)" % (KNN_FD_OH_mean, KNN_FD_OH_variance))
