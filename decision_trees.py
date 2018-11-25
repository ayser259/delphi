from sklearn import preprocessing, tree
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score

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
clf = tree.DecisionTreeClassifier()
cvs = cross_val_score(clf,x_df,y_df,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
