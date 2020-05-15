# -*- coding: utf-8 -*-
# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization
import numpy as np
from numpy import array
import itertools
from util import UTIL
import math
import random


class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=10, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_Item = None  # Item feature vectors
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

        pairs_train = train_vec.shape[0]  # traindata 中条目数
        pairs_test = test_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数

        incremental = False  # 增量
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  # 检查迭代次数
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):  # 每次迭代要使用的数据量
                # print "epoch %d batch %d" % (self.epoch, batch+1)

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply对应位置元素相乘

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda * (np.linalg.norm(self.w_User) ** 2 + np.linalg.norm(self.w_Item) ** 2)

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))
        print(self.w_Item.shape)
        print(self.w_User.shape)
        
    def predict(self, invID):
        return np.dot(self.w_Item, self.w_User[int(invID), :])+ self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0]) # unique users
        pred = {}

        mrr = 0
        positive_pred = {}

        for inv in inv_lst: 
            if pred.get(inv, None) is None: 
                pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序 # argsort returns indexes that would sort the array, index is a movie
                                                           # we take the k last indexes: the best k movies
                positive_pred[inv] = []     

        
        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]: # for each rating if the movie is in the movie list of predictions for the user
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1 # we increment, second argument of get is default value
                positive_pred[test_vec[i,0]].append(test_vec[i,1])
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32')) # counts the number of ratings for each user

        for inv in inv_lst: 
            ranked = False
            for movie in pred[inv]:
                if(movie in positive_pred[inv]):
                    if(ranked == False):
                        ranked = True
                        mrr += 1/(np.where(pred[inv]==movie)[0][0]+1)

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
        return precision_acc / len(inv_lst), recall_acc / len(inv_lst), 2*(precision_acc/len(inv_lst))*(recall_acc/len(inv_lst))/((precision_acc/len(inv_lst))+(recall_acc/len(inv_lst))),mrr/len(inv_lst)

    def fitBPR(self, train_vec, test_vec, threshold, lambda_user, lambda_pos, lambda_neg):

        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值
        
        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数
        
        if (self.w_Item is None):
            self.w_Item =  0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_User =  0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

        """
        self.w_Item.round(3, out=self.w_Item)
        self.w_User.round(3, out=self.w_User)
        self.w_Item = np.array(self.w_Item, dtype=np.float64)
        self.w_User = np.array(self.w_User, dtype=np.float64)

     
        for i in range(len(self.w_Item)):
            for j in range(len(self.w_Item[i])):
                self.w_Item[i][j] = int(self.w_Item[i][j]*1000)/1000

        for i in range(len(self.w_User)):
            for j in range(len(self.w_User[i])):
                self.w_User[i][j] = int(self.w_User[i][j]*1000)/1000

        print(max([max(p) for p in self.w_Item]))
        print(min([min(p) for p in self.w_Item]))
        """
        triples = self.constructTriples(train_vec,num_user,num_item,threshold)

        util = UTIL()

        random.shuffle(triples)
       
        for index in range(len(triples)):
            
            vi = self.w_Item[int(triples[index][1])].round(3, out = self.w_Item[int(triples[index][1])])
            vj = self.w_Item[int(triples[index][2])].round(3, out = self.w_Item[int(triples[index][2])])
            u =  self.w_User[int(triples[index][0])].round(3, out = self.w_User[int(triples[index][0])])
            
            x_uij = np.dot(u,vi-vj)
            #x_uij = np.sum(u * (vi - vj), axis = 0)
            #x_uij = util.dot_product(vi,u) - util.dot_product(vj,u)
           
            #x_uij = util.dot_product(self.w_Item[int(triples[index][1])],self.w_User[int(triples[index][0])]) - util.dot_product(self.w_Item[int(triples[index][2])],self.w_User[int(triples[index][0])])
            #x_uij = np.dot(self.w_Item[int(triples[index][1])],self.w_User[int(triples[index][0])]) - np.dot(self.w_Item[int(triples[index][2])],self.w_User[int(triples[index][0])])
            
            """
            s_user = np.square(self.w_User[int(triples[index][0])])
            ms_user = np.mean(s_user)
            rms_user = np.sqrt(ms_user)

            s_pos = np.square(self.w_Item[int(triples[index][1])])
            ms_pos = np.mean(s_pos)
            rms_pos = np.sqrt(ms_pos)

            s_neg = np.square(self.w_Item[int(triples[index][2])])
            ms_neg = np.mean(s_neg)
            rms_neg = np.sqrt(ms_neg)
            """
            sigmoid = np.exp(-x_uij) / (1.0 + np.exp(-x_uij))
            sigmoid_tiled = np.tile(sigmoid, (self.num_feat, 1)).T

            theta_user = self.epsilon *( sigmoid_tiled*(vj - vi )  + lambda_user*u)
            theta_pos = self.epsilon*(  sigmoid_tiled*(-u) +lambda_pos*vi)
            theta_neg = self.epsilon*( sigmoid_tiled*(u) +lambda_neg*vj)
            #lambda x: .5 * (math.tanh(.5 * x) + 1)
            """
            theta_user = self.epsilon *( (math.exp(-x_uij)/(1 + math.exp(-x_uij)))*(vi - vj )  + lambda_user*u)
            theta_pos = self.epsilon*(  (math.exp(-x_uij)/(1 + math.exp(-x_uij)))*(u) +lambda_pos*vi)
            theta_neg = self.epsilon*( (math.exp(-x_uij)/(1 + math.exp(-x_uij)))*(-u) +lambda_neg*vj)
            """
            #theta_user = self.epsilon*( (1 - 0.5 * (math.tanh(.5 * -x_uij) + 1))*(self.w_Item[int(triples[index][1])]-self.w_Item[int(triples[index][2])])+self._lambda*rms_user)
            #theta_pos = self.epsilon*(  (1- .5 * (math.tanh(.5 * -x_uij) + 1))*(self.w_User[int(triples[index][0])])+self._lambda*rms_pos)
            #theta_neg = self.epsilon*( (1- .5 * (math.tanh(.5 * -x_uij) + 1))*(-self.w_User[int(triples[index][0])])+self._lambda*rms_neg)
            """
            theta_user = self.epsilon*( (1 - 1/(1+np.exp(-x_uij)))*(self.w_Item[int(triples[index][1])]-self.w_Item[int(triples[index][2])])+self._lambda*rms_user)
            theta_pos = self.epsilon*(  (1- 1/(1+np.exp(-x_uij)))*(self.w_User[int(triples[index][0])])+self._lambda*rms_pos)
            theta_neg = self.epsilon*( (1- 1/(1+np.exp(-x_uij)))*(-self.w_User[int(triples[index][0])])+self._lambda*rms_neg)
            """
            """
            #without RMS
            theta_user = self.epsilon*( (1 - 1/(1+np.exp(-x_uij)))*(self.w_Item[int(triples[index][1])]-self.w_Item[int(triples[index][2])])+self._lambda*self.w_User[int(triples[index][0])])
            theta_pos = self.epsilon*( (1 - 1/(1+np.exp(-x_uij)))*(self.w_User[int(triples[index][0])])+self._lambda*self.w_Item[int(triples[index][1])])
            theta_neg = self.epsilon*( (1 - 1/(1+np.exp(-x_uij)))*(-self.w_User[int(triples[index][0])])+self._lambda*self.w_Item[int(triples[index][2])])
            """
            #sometimes exp(-x_uij) overflows
            #if(len(np.argwhere(np.isnan(theta_user)))==0 ):
             #   counter += 1
            self.w_User[int(triples[index][0])] = self.w_User[int(triples[index][0])] - theta_user
            #if(len(np.argwhere(np.isnan(theta_pos)))==0 ):
             #   counter += 1
            self.w_Item[int(triples[index][1])] = self.w_Item[int(triples[index][1])] - theta_pos
            #if(len(np.argwhere(np.isnan(theta_neg)))==0 ):
            self.w_Item[int(triples[index][2])] = self.w_Item[int(triples[index][2])] - theta_neg
            if(index%100000 == 0):
                print("Iteration "+ str(index)+" PMF-BPR: precision_acc,recall_acc,F1,MRR:" + str(self.topK(test_vec)))

    def sigmoid(self,gamma):
        if gamma < 0:
            return round(1 - 1/(1 + math.exp(round(gamma,3))),3)
        else:
            return round(1/(1 + math.exp(round(-gamma,3))),3)

    def constructTriples(self, train_vec, num_user, num_item, threshold):
        """
        hits_matrix = np.zeros((num_user, num_item))  # used to identify positive and negative items
        for i in range(len(train_vec)):
            hits_matrix[int(train_vec[i][0])][int(train_vec[i][1])] = 1
        """

        positive = {}
        negative = {}

        for user in range(1,num_user):
            positive[user] = []
            negative[user] = []

        for i in range(len(train_vec)):
            if(train_vec[i,2]>=threshold):
                positive.get(train_vec[i,0]).append(train_vec[i,1])
            else:
                negative.get(train_vec[i,0]).append(train_vec[i,1])

        triples = []
        for user in range(1,num_user):
            user_triples = []
            for combination in itertools.product([user], positive.get(user), negative.get(user)):
                user_triples.append(list(combination))
            triples += user_triples
        
        print("Triples Constructed")

        return triples

        




