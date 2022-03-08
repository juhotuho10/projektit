import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# for getting data from the website
import requests

# data location
Minimum_wage_csv_location = "C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_science-Pandas-2/Minimum Wage Data.csv"

# file save locations
State_abbv_csv_location = "C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_science-Pandas-2/state_abbv.csv"
Min_wage_UTF8_location = 'C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_science-Pandas-2/MinimumWage-UTF8.csv'

# convert latin -> utf-8
df = pd.read_csv(
    Minimum_wage_csv_location,
    encoding='latin')
df.to_csv(
    Min_wage_UTF8_location,
    encoding='utf-8')

print(df.head())

# gets a group of certain states
gb = df.groupby('State')
print(gb.get_group('Alabama').set_index('Year').head())

act_min_wage = pd.DataFrame()

for name, group in df.groupby('State'):  # name = state, group = data
    if act_min_wage.empty:  # makes a new column, takes the data from column low.2018 and renames it
        act_min_wage = group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name})
    else:
        act_min_wage = act_min_wage.join(group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name}))

print(act_min_wage.head())  # Columns of states with yearly min.wage

print(act_min_wage.describe())  # data info(mean, div, min, max)

print(act_min_wage.corr())  # correlation of  the data

issue_df = df[df['Low.2018'] == 0]  # df missä min.wage =0 0
print(issue_df.head())
print(len(issue_df['State'].unique()))  # 15 osavaltiosta ei ole dataa

# act_min_wage.replace(0, np.NaN).dropna(axis=1) #replaces 0s with NaN values, then gets rid of them
# axis = 1 gets  rid of the row, 0 would be column
# but if the state has had a time of no min.wage? can't del those

grouped_issues = issue_df.groupby('State')
print(grouped_issues.get_group('Alabama').head())

print(grouped_issues.get_group('Alabama')['Low.2018'].sum())  # prints state minimum wage sum

for state, data in grouped_issues:
    if data['Low.2018'].sum() != 0.0:
        print(f'This state doesnt have missing data:{state}')

# all the problem states have missing data so we can just del them
min_wage_corr = act_min_wage.replace(0, np.NaN).dropna(axis=1).corr()
print(min_wage_corr.head())  # no more NaN

# Getting better labels than 2 first letters from the net
web = requests.get('https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes')

dfs = pd.read_html(web.text)  # shows the text displayed in the website (returns a list of dataframes)

'''for df in dfs:
	print(df.head())'''

state_abbv = dfs[0]

# save the data
state_abbv.to_csv(
    State_abbv_csv_location,
    encoding='utf-8', index=False)

# load the data from the save
state_abbv = pd.read_csv(
    State_abbv_csv_location,
    index_col=0)

# print(state_abbv.head())

# takes postal codes and convers them to dictionary
abbv_dict = state_abbv[['Postal Code']].to_dict()

# state_abbv where key == 'postal code'

abbv_dict = abbv_dict['Postal Code']
# adding some missing values
abbv_dict['Federal (FLSA)'] = "FLSA"
abbv_dict['Guam'] = "GU"
abbv_dict['Puerto Rico'] = "PR"

print(min_wage_corr.columns)
print(abbv_dict)

labels = [abbv_dict[c] for c in min_wage_corr.columns]
print(state_abbv)

# pyplot customization

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

# color map = red yellow green
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

plt.show()
