from data_load import get_clean_data,get_one_hot_encoded_data,merged_encoding

betty = get_clean_data('data.csv')
df = get_one_hot_encoded_data('data.csv',drop_pref=True)

merged_encoded = merged_encoding('data.csv',['hs_average','social_time'],['screen_time','sleep_time'],drop_pref=True,)

print(merged_encoded)
