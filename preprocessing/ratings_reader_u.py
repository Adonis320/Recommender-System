import pandas as pd
import csv

# splits dataset and creates a binarized version

data = pd.read_csv('../data/dataFormatted.csv')

randomizedData = data.sample(frac=1)

#80% training data, 20% test
train_data = randomizedData[:80000]
#validation_data = randomizedData[80000:100000]
test_data = randomizedData[80000:100000]

#export to csv
train_data.to_csv(r'../data/training/ratings.csv', index = False)
#validation_data.to_csv(r'../data/validation/ratings.csv', index = False)
test_data.to_csv(r'../data/test/ratings.csv', index = False)

#threshold for binarization
threshold = 4
#binarization of ratings: if value >= threshold then 1 else 0
train_data["rating"] = (train_data["rating"] >= threshold).astype(int)
#validation_data["rating"] = (validation_data["rating"] >= threshold).astype(int)
test_data["rating"] = (test_data["rating"] >= threshold).astype(int)

#export binarized to csv
train_data.to_csv(r'../data/training/binarized/ratings.csv', index = False)
#validation_data.to_csv(r'../data/validation/binarized/ratings.csv', index = False)
test_data.to_csv(r'../data/test/binarized/ratings.csv', index = False)