import pandas as pd
import random

#read csv file into pandas.dataFrame
data = pd.read_csv('../data/raw/ratings.csv')

#show the data
print(data)

#check for duplicates
duplicateRowsDF = data[data.duplicated()]
#print(duplicateRowsDF)

#returns the numpy representation of the dataFrame
#values = data.to_numpy()
#print(values)
#prints the value at line i column j-1
#print(values[0][2])

#shuffles the data
randomizedData = data.sample(frac=1)

print(randomizedData)

#80% training data, 10% validation, 10% test
train_data = randomizedData[:70586]#[:80669]
validation_data = randomizedData[70586:100836]#[80669:90753]
test_data = randomizedData[90753:100836]

#export to csv
train_data.to_csv(r'../data/training/ratings.csv', index = False)
validation_data.to_csv(r'../data/validation/ratings.csv', index = False)
test_data.to_csv(r'../data/test/ratings.csv', index = False)

#threshold for binarization
threshold = 4
#binarization of ratings: if value >= threshold then 1 else 0
train_data["rating"] = (train_data["rating"] >= threshold).astype(int)
validation_data["rating"] = (validation_data["rating"] >= threshold).astype(int)
test_data["rating"] = (test_data["rating"] >= threshold).astype(int)

#export binarized to csv
train_data.to_csv(r'../data/training/binarized/ratings.csv', index = False)
validation_data.to_csv(r'../data/validation/binarized/ratings.csv', index = False)
test_data.to_csv(r'../data/test/binarized/ratings.csv', index = False)

