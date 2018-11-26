import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from data_load import get_encoded_data

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

mean = np.array(accuracy).mean()
variance = np.array(accuracy).std() * 2
print("LOO CV Accuracy: %0.2f (+/- %0.2f)" % (mean, variance))
