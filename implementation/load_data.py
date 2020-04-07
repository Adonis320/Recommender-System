# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization
from numpy import *
from numpy import array
import random

def load_training_data():
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """

    #import training data
    prefer = []
    for line in open('../data/training/ratings.csv', 'r').readlines()[1:]:  # 打开指定文件 # we skip the first line because of the header
        (userid, movieid, rating) = line.split(',')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    return data

def load_test_data():
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """

    #import training data
    prefer = []
    for line in open('../data/test/ratings.csv', 'r').readlines()[1:]:  # 打开指定文件  #we skip the first line because of the header
        (userid, movieid, rating) = line.split(',')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    return data