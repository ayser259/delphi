from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score

from data_load import get_encoded_data

df = get_encoded_data('data.csv')[0]
# encoded_dict_list = get_encoded_data('data.csv')[1]

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

# Cross validation test train split
clf = KNeighborsClassifier(n_neighbors=3).fit(x_df,y_df)
cvs = cross_val_score(clf,x_df,y_df,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
