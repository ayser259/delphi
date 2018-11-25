from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from data_load import get_clean_data

df = get_clean_data('data.csv')

# gnb = MultinomialNB()

x_df = df.drop(axis=1,columns=["current_average"])
y_df = df["current_average"]
