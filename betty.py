from data_load import get_clean_data,get_one_hot_encoded_data

betty = get_clean_data('data.csv')
df = get_one_hot_encoded_data('data.csv',drop_pref=True)
print(df)
