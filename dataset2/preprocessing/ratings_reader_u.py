# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization

import pandas as pd
import csv

prefer = []
for line in open('../data/u.txt', 'r'): 
        (userid, movieid, rating, ts) = line.split('\t') 
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        lin = [uid,mid,rat]
        prefer.append(lin)
df = pd.DataFrame(prefer)
df.columns = ["userid","movieid","rating"]
df.to_csv(r'../data/dataFormatted.csv', index = False, quoting=csv.QUOTE_NONE,escapechar=" ")