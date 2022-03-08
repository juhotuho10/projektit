import pandas as pd 



df = pd.read_csv('C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_Science-Pandas-1/avocado.csv')

print(df.head())

#prints columns
for i in df:
	print(i)

#albany_df == where df region column == albany
albany_df = df[df['region'] == 'Albany']
print(albany_df.head())

#index = data number
albany_df = albany_df.set_index('Date') #albany_df.set_index('Date', inplace=True)
print(albany_df.head())

