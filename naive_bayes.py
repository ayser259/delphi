from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score

from data_load import get_clean_data

df = get_clean_data('data.csv')

col_list = list(df.columns)
encoded_dict_list = []
for col in col_list:
    if col!= "Timestamp":
        keys = df[col].unique()
        le = preprocessing.LabelEncoder()
        le.fit(list(keys))
        df[col] = le.transform(list(df[col]))
        vals = df[col].unique()
        keys = list(le.inverse_transform(vals))
        cd = dict(zip(keys,vals))
        cd['column'] = col
        encoded_dict_list.append(cd)

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]

# Basic train test split:
'''
x_train,x_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.1)

mnb = MultinomialNB()
mnb.fit(x_train,y_train)

y_predicted = mnb.predict(x_test)
print(accuracy_score(y_test,y_predicted))
'''
# Cross validation test train split
mnb = MultinomialNB()
cvs = cross_val_score(mnb,x_df,y_df,cv=50)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
