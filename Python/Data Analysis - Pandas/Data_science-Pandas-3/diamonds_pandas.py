import pandas as pd
import sklearn
from sklearn import svm, preprocessing

data_csv_location = 'C:/Users/TK-Kone/Documents/Python hommelit/Data Analysis - Pandas/Data_science-Pandas-4/diamonds.csv'

df = pd.read_csv(data_csv_location,
                 index_col=0)
print(df.head())  # we want to predict price

# convert everything to numbers!!
print(df['cut'].unique())

'''
#assigns an arbitrary code to every new piece in category it finds (first = 1, second = 2)
#not necessarily in order (worst -> best) so we won't use it
print(df['cut'].astype('category').cat.codes)
'''

# assign key value pairs to replace the current values
cut_class_dict = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
clarity_dict = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8, "VVS1": 9, "IF": 10,
                "FL": 11}
color_dict = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}

# mapping the values according to the keys
df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)

print(df.head())

# shuffle the data
df = sklearn.utils.shuffle(df)
print(df.head())

# create the training data X and validation data y
X = df.drop('price', axis=1).values
y = df['price'].values

# scale the data to simplify it
X = preprocessing.scale(X)

test_size = 500

# training data up to test size
X_train = X[:-test_size]
y_train = y[:-test_size]

# test data from test size and up
X_test = X[-test_size:]
y_test = y[-test_size:]

# def classifier
clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)

# evaluates the model
print(clf.score(X_test, y_test))
