import numpy as np
import pandas as pd
from sklearn import preprocessing,tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

from data_load import get_encoded_data

df = get_encoded_data('data.csv')[0]
# encoded_dict_list = get_encoded_data('data.csv')[1]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

# Basic train test split:
'''
x_train,x_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.1)

clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)

y_predicted = clf.predict(x_test)
print(accuracy_score(y_test,y_predicted))
'''
# Cross validation test train split
'''
clf = tree.DecisionTreeClassifier()
cvs = cross_val_score(clf,x_df,y_df,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
'''
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

  # TODO: this is where you change it to the specific algorithim: i.e. KNN, naives bayes, decision trees
  ent = tree.DecisionTreeClassifier()
  model = ent.fit(X_train, y_train) # fit the model using training data
  accuracy.append(ent.score(X_test, y_test))
  
# Calculate accuracy
mean = np.array(accuracy).mean()
variance = np.array(accuracy).std() * 2
print("LOO CV Accuracy: %0.2f (+/- %0.2f)" % (mean, variance))
