import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.naive_bayes import MultinomialNB

from data_load import get_encoded_data,get_one_hot_encoded_data

# Label Encoded Data
df = get_encoded_data('data.csv')[0]

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

  mnb = MultinomialNB()
  model = mnb.fit(X_train, y_train) # fit the model using training data
  accuracy.append(mnb.score(X_test, y_test))

# Calculate accuracy
label_mean = np.array(accuracy).mean()
label_variance = np.array(accuracy).std() * 2

# One Hot Encoded
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

  mnb = MultinomialNB()
  model = mnb.fit(X_train, y_train) # fit the model using training data
  accuracy.append(mnb.score(X_test, y_test))

# Calculate accuracy
ohe_mean = np.array(accuracy).mean()
ohe_variance = np.array(accuracy).std() * 2

#Printing Results
print("Label Encoded LOO CV Accuracy: %0.2f (+/- %0.2f)" % (label_mean, label_variance))
print("One Hot Encoded LOO CV Accuracy: %0.2f (+/- %0.2f)" % (ohe_mean, ohe_variance))
