from data_load import get_clean_data,get_one_hot_encoded_data,merged_encoding

betty = get_clean_data('data.csv')
df = get_one_hot_encoded_data('data.csv',drop_pref=True)

merged_encoded = merged_encoding('data.csv',['hs_average','social_time'],['screen_time','sleep_time'],drop_pref=True,)

print(merged_encoded)


# Basic train test split:
'''
x_train,x_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.1)

mnb = MultinomialNB()
mnb.fit(x_train,y_train)

y_predicted = mnb.predict(x_test)
print(accuracy_score(y_test,y_predicted))
'''
# Cross validation test train split
'''
mnb = MultinomialNB()
cvs = cross_val_score(mnb,x_df,y_df,cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std() * 2))
'''
