from data_load import get_clean_data
import csv
import pandas as pd
import numpy as np
import mlxtend.preprocessing as mlx
from mlxtend.preprocessing import transactionencoder as mlx
from mlxtend.frequent_patterns import apriori, association_rules

df = get_clean_data("data.csv")
# df = pd.read_csv("data.csv", usecols=list(range(1,15)))

# print(df)
df['hs_average'] = 'hs_avg_' + df['hs_average'].astype(str)
df['current_year'] = 'curr_yr_' + df['current_year'].astype(str)
df['faculty'] = 'faculty_' + df['faculty'].astype(str)
df['nationality_status'] = 'nationality_status_' + df['nationality_status'].astype(str)
df['parent1_education'] = 'parent1_education_' + df['parent1_education'].astype(str)
df['parent2_education'] = 'parent2_education_' + df['parent2_education'].astype(str)
df['social_time'] = 'social_time_' + df['social_time'].astype(str)
df['class_attendance'] = 'class_attendance_' + df['class_attendance'].astype(str)
df['screen_time'] = 'screen_time_' + df['screen_time'].astype(str)
df['sleep_time'] = 'sleep_time_' + df['sleep_time'].astype(str)
df['excercise_time'] = 'excercise_time_' + df['excercise_time'].astype(str)
df['school_work_time'] = 'school_work_time_' + df['school_work_time'].astype(str)
df['coop_time'] = 'coop_time_' + df['coop_time'].astype(str)
df['academic_priority'] = 'academic_priority_' + df['academic_priority'].astype(str)
df['current_average'] = 'current_average_' + df['current_average'].astype(str)


df.to_csv("output.csv",header=False,index=False, columns=['hs_average','nationality_status','parent1_education','parent2_education','social_time','screen_time','sleep_time','excercise_time','school_work_time','coop_time','academic_priority','current_average'])



# Reads the output.csv created from above and puts the data into a list of lists
dataset = []
with open('output.csv') as f:
    reader = csv.reader(f)
    for row in reader: 
        dataset.append(row)


# Creates a dataframe with the row being each student, and the columns being
# every single possibile value with true or false depending if the user falls into that category
oht = mlx.TransactionEncoder()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)

# Creates a set of items with a min support as specified by the min_support param
frequent_itemSets = apriori(df,min_support=0.03, use_colnames=True)

# Only looks at the association rules with more then 2 items
frequent_itemSets['length'] = frequent_itemSets['itemsets'].apply(lambda x: len(x))
frequent_itemSets[ (frequent_itemSets['length'] >= 2) ]
# print(frequent_itemSets[ (frequent_itemSets['length'] >= 2) ])

#Creates rules based off the min_threshold
rules = association_rules(frequent_itemSets, metric="confidence", min_threshold=0.95)



print(rules[['antecedents','consequents','support','confidence']])
rules[['antecedents','consequents','support','confidence']].to_csv("apriori-output.csv")




