from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from data_load import get_encoded_data, get_clean_data, get_one_hot_encoded_data

# df = get_one_hot_encoded_data('data.csv', drop_pref=True)
df = encoded_dict_list = get_encoded_data('data.csv')[0]
encoded_dict_list = get_encoded_data('data.csv')[1]
# print(get_encoded_data('data.csv'))

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

## USING SCLEARN TO TEST AND TRAIN
TEST_SIZE = 0.5
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=TEST_SIZE)
KNN = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
print("Accuracy: ", KNN.score(X_test, y_test))
print("test: ", KNN.predict(X_test))


## ===== LEAVE ONE OUT CROSS VALIDATION BEGINS HERE ==== ##
X = np.array(x_df) # convert dataframe into np array
y = np.array(y_df) # convert dataframe into np array

loo = LeaveOneOut()
loo.get_n_splits(X)
LeaveOneOut()

accuracy = []

for train_index, test_index in loo.split(X):
  X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index]) # use this for training the model
  y_train, y_test = y[train_index].ravel(), y[test_index].ravel() # use this for testing the model

  # TODO: this is where you change it to the specific algorithim: i.e. KNN, naives bayes, decision trees
  KNN = KNeighborsClassifier(n_neighbors=3)
  model = KNN.fit(X_train, y_train) # fit the model using training data
  accuracy.append(KNN.score(X_test, y_test))

# Calculate accuracy
mean = np.array(accuracy).mean()
variance = np.array(accuracy).std() * 2
print("LOO CV Accuracy: %0.2f (+/- %0.2f)" % (mean, variance))
