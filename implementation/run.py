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
    train_data = load_training_data()
    test_data = load_test_data()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})

    #pmf.fit(train_data, test_data)
    pmf.fitBPR(train_data,test_data,4,1,1,1)

    inv_lst = np.unique(test_data[:, 0]) # users list in the test data
    
    pred = {} # movies predicted from PMF
    for inv in inv_lst: 
        movies = []
        if pred.get(inv, None) is None: 
            pred[inv] = np.argsort(pmf.predict(inv))
            pred[inv] = pred.get(inv)[::-1] # ranked movies from best to worst

    util = UTIL()

    similarity_matrix = [] # similarity matrix between movies
    similarity_with_users = {} # similarity matrixes between each user and all movies
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
    mmr_pred = {}
    for i in range(1,len(inv_lst)): 
        if(pred.get(i) is not None): # some users don't exist in the test data
            mmr = MMR(pred.get(i),i,similarity_matrix,similarity_with_users,1,k) 
            mmr_pred[i] = mmr.rank()
    
    util.evaluateMMR(mmr_pred,test_data,10,inv_lst)

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
    print("PMF: precision_acc,recall_acc,F1,MRR:" + str(pmf.topK(test_data)))
