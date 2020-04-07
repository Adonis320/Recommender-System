# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from probabilistic_matrix_factorization import PMF
from load_data import load_training_data, load_test_data
from mmr import MMR
from util import UTIL

if __name__ == "__main__":

    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
    train_data = load_training_data()
    test_data = load_test_data()
    pmf.fit(train_data, test_data)

    inv_lst = np.unique(test_data[:, 0]) # users list in the test data
    
    pred = {} # movies predicted from PMF
    for inv in inv_lst: 
        movies = []
        if pred.get(inv, None) is None: 
            pred[inv] = np.argsort(pmf.predict(inv))
            pred[inv] = pred.get(inv)[::-1] # ranked movies from best to worst

    util = UTIL()

    similarity_matrix = []
    similarity_with_users = {}
    for m1 in pmf.w_Item:
        sim = []
        for m2 in pmf.w_Item:
            sim.append(util.compute_similarity(m1,m2))
        similarity_matrix.append(sim)    
    print("Similarity Matrix calculated")

    for inv in inv_lst:
        similarity_with_users[int(inv)] = []
        for m1 in pmf.w_Item:
            similarity_with_users.get(int(inv)).append(util.compute_similarity(m1,pmf.w_User[int(inv),:]))
    print("Similarity with users Calculated")


    


    k = 10 # number of movies returned by MMR
    new_pred = {}
    for i in range(1,len(inv_lst)): 
        if(pred.get(i) is not None): # some users don't exist in the test data
            mmr = MMR(pred.get(i),i,similarity_matrix,similarity_with_users,1,k) 
            new_pred[i] = mmr.rank()

    intersection_cnt = {}
    for i in range(test_data.shape[0]):
        if new_pred.get(int(test_data[int(i), 0])) is not None:
            if (int(test_data[int(i), 1])) in (new_pred[int(test_data[int(i), 0])]): # for each rating if the movie is in the movie list of predictions for the user
                intersection_cnt[test_data[int(i), 0]] = intersection_cnt.get(test_data[int(i), 0], 0) + 1 # we increment, second argument of get is default value
    invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32')) # counts the number of ratings for each user
    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
    print("MMR: precision_acc = " +str(precision_acc / len(inv_lst))+" recall_acc ="+str(recall_acc / len(inv_lst)))
    
    #TODO for the precision comparison number of users for MMR is different than the number of users used by PMF topK

    """
    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    """
    print("PMF: precision_acc,recall_acc:" + str(pmf.topK(test_data)))
