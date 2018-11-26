import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from data_load import get_encoded_data

df = get_encoded_data('data.csv')[0]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

# Cross validation test train split
'''
clf = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='multinomial').fit(x_df, y_df)
cvs = cross_val_score(clf,x_df,y_df,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
'''

X = np.array(x_df) # convert dataframe into np array
y = np.array(y_df) # convert dataframe into np array

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  LRR = LogisticRegression(random_state=0, solver='lbfgs',
                              multi_class='multinomial')
  model = LRR.fit(X_train, y_train) # fit the model using training data
  accuracy.append(LRR.score(X_test, y_test))

# Calculate accuracy
mean = np.array(accuracy).mean()
variance = np.array(accuracy).std() * 2
print("LOO CV Accuracy: %0.2f (+/- %0.2f)" % (mean, variance))
