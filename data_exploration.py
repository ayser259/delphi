## 1 ## Code block

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import PercentFormatter

## 2 ## Code block

sns.set(style="darkgrid")
sns.set(style="ticks", color_codes=True)

## 3 ## Code block

# Reading data from the data source
data = pd.read_csv('data.csv')
# Changing column names for readability
column_dict = {
    "What year are you in?": "current_year",
    "What faculty are you in?": "faculty",
    "What was your high school average when you applied to the University of Waterloo?": "hs_average",
    "What is your nationality status?": "nationality_status",
    "What is the highest education level your parents have completed?  [Parent 1]": "parent1_education",
    "What is the highest education level your parents have completed?  [Parent 2]": "parent2_education",
    "On average, how much time do you spend per week participating in social activities during an academic term? (i.e. extracurricular activities, movies, eating out, bars, parties, hanging out, etc.)": "social_time",
    "On average, what percentage of classes do you feel that you attend during an academic term? ": "class_attendance",
    "On average, excluding studying, how much time do you spend looking at a screen, during an academic term? (i.e. Phone, Laptop, TV, etc) ": "screen_time",
    "On average, how much sleep do you get per night during an academic term? ": "sleep_time",
    "On average, how many days do you exercise each week during an academic term?": "excercise_time",
    "On average, how much time do you spend doing school work / studying on a given day during an academic term?": "school_work_time",
    "In relation to school work, how much time do you spend on coop preparation during an academic term? (ie. applications. interview prep. practice, interviews, etc)": "coop_time",
    "Is it a high priority for you to achieve an 80%+ average": "academic_priority",
    'What is your current cumulative average?': "current_average"
}
#  Dictionary for Changing nationality answers for readability
nationality_dict = {
    'International (You are not a Canadian Citizen and are here on a Visa)':'Internationl',
    '1st Generation Canadian Citizen (You were not born in Canada and You are a Canadian Citizen)':'1st_Gen',
    '2nd+ Generation Canadian Citizen (You were born in Canada and you are a Canadian Citizen)':'2nd+_Gen'
}
#  Dictionary for Changing sleep time answers for readability
sleep_dict = {
    "Not enough (I'm always tired)":"Not Enough",
    "Enough (I'm rested most of the time)":"Enough",
    "More than enough (I'm always well rested)":"More than Enough"
}
#  Dictionary for Changing screen time answers for readability
screen_dict = {
    'Regularly, but not a significant amount':'Regularly',
    "I'm almost always looking at a screen":'Almost Always',
    'A significant amount':'A significant amount',
    'Almost never':'Almost never'
}
#  Dictionary for Changing co-op time answers for readability
coop_dict = {
    'Almost none':'Almost none',
    'About the same if not more as school work':"Same or More",
    'A significant amount but still less than school work':'Significant,but less than school',
    'A lot less than school work':'A lot less than school work'
}
# Dictionary for Changing social time answers for readability
social_dict = {
    'Once or twice a week':'Once/Twice Weekly',
    'Multiple days a week':'Multiple Weekly',
    'Rarely':'Rarely'
}
# Renaming columns
data = data.rename(index=str,columns = column_dict)
# Updating a few rows of data due to incorrect initial survey deployment
for i in range(0,12):
    data.faculty[i]='Engineering'
# Extracting data for only engineering students
data = data[data['faculty']=="Engineering"]

# Updating a few rows of data due to incorrect initial survey deployment
for i in range(0,12):
    data.loc["faculty",i]='Engineering'
# Dropping e-mail address values for confidentiality reasobs
data = data.drop(axis=1,columns=["Enter your email address OR phone number if you'd like to be entered for a chance to win 1 of 4 $20 amazon gift cards"])
# Updating column values for readability
data.nationality_status = data.nationality_status.map(nationality_dict)
data.sleep_time = data.sleep_time.map(sleep_dict)
data.social_time = data.social_time.map(social_dict)
data.coop_time = data.coop_time.map(coop_dict)
data.screen_time = data.screen_time.map(screen_dict)

## 4 ## Code block

# Defining methods to help normaliz the data
def normalize_3_variables(df3,x,y,column,hue):
    normalized_data = df3[[x,y,column,hue]]
    normalized_data = normalized_data.groupby([x,y,column,hue],as_index=False).size().reset_index()
    normalized_data = normalized_data.rename(index=str,columns = {0:"percent"})
    normalized_data["percent"] = 100*(normalized_data["percent"]/sum(normalized_data["percent"]))
    return normalized_data

def normalize_2_variables(df2,x,y,column):
    normalized_data = df2[[x,y,column]]
    normalized_data = normalized_data.groupby([x,y,column],as_index=False).size().reset_index()
    normalized_data = normalized_data.rename(index=str,columns = {0:"percent"})
    normalized_data["percent"] = 100*(normalized_data["percent"]/sum(normalized_data["percent"]))
    return normalized_data

def normalize_1_variables(df1,x,y):
    normalized_data = df1[[x,y]]
    normalized_data = normalized_data.groupby([x,y],as_index=False).size().reset_index()
    normalized_data = normalized_data.rename(index=str,columns = {0:"percent"})
    normalized_data["percent"] = 100*(normalized_data["percent"]/sum(normalized_data["percent"]))
    return normalized_data


## 5 ## Code Block

# Objects to help clean and structure the data

order=['60-64%','65-69%','70-74%','75-79%','80-84%','85-89%','90-94%','95-100%''Prefer not to say']

## 6 ## Markdown Block

# Exploring Relationship of cGPA with All Explanatory Variables

Initial exploratory data analysis is being conducted by through basic charts in order to get a better understanding of all of the existing data, in order to see the distribution of the data. This is being done as an altnernative to generating summary tables of the existing data, due to the variance of data types accross columns. The relationship of each of the variables with the cGPA are explored below.
