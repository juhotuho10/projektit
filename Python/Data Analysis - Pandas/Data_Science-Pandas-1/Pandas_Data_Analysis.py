import pandas as pd
from matplotlib import pyplot as plt

Data_csv_location = 'C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_Science-Pandas-1/avocado.csv'

df = pd.read_csv(Data_csv_location)

df = df.copy()[df['type'] == 'organic']  # removes duplicate dates for same region for different type
df['Date'] = pd.to_datetime(df['Date'])  # converts the date column to a date format for pandas

df.sort_values(by='Date', ascending=True, inplace=True)  # sort by Date column

albany_df = df.copy()[df['region'] == 'Albany']
albany_df.set_index('Date', inplace=True)

# from Avg.Price column averages out 25 columns values to their mean (avg.)
albany_df['AveragePrice'].rolling(25).mean()
print(albany_df.index)  # starts at 2015, goes down, ends in 2018 :think:

# albany_df['AveragePrice'].rolling(25).mean().plot()
# plt.show()

albany_df.sort_index(inplace=True)  # sorts the data by index (Date)
albany_df['price25mean'] = albany_df['AveragePrice'].rolling(25).mean()  # creating a new column for 25 average

# albany_df['price25mean'].plot()
# plt.show()


'''regions = df['region']
regions = regions.values.tolist()
regions = set(regions)
regions = list(regions)
print(regions)'''

print(df['region'].unique())  # sama asia

graph_df = pd.DataFrame()

for region in df['region'].unique():
    print(region)
    region_df = df.copy()[df['region'] == region]
    region_df.set_index('Date', inplace=True)
    region_df.sort_index(inplace=True)
    region_df[f'{region}_price25mean'] = region_df['AveragePrice'].rolling(25).mean()

    if graph_df.empty:
        graph_df = region_df[[f'{region}_price25mean']]  # takes the price 25 mean column in its place

    else:
        graph_df = graph_df.join(region_df[f'{region}_price25mean'])

print(graph_df)

graph_df.dropna().plot(figsize=(10, 7), legend=False)
plt.show()
