# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from probabilistic_matrix_factorization import PMF
from load_data import load_training_data, load_validation_data, load_test_data
from mmr import MMR

if __name__ == "__main__":

    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
    train_data = load_training_data()
    validation_data = load_validation_data()
    test_data = load_test_data()
    
    pmf.fit(train_data, validation_data)

    inv_lst = np.unique(validation_data[:, 0]) # 
    pred = {}
    recommended_movies_properties = {}

    k = 30
    for inv in inv_lst: 
        movies = []
        if pred.get(inv, None) is None: 
            pred[inv] = np.argsort(pmf.predict(inv))#[-k:]
            pred[inv] = pred.get(inv)[::-1] # ranked movies from best to worst
            for movie in pred.get(inv):
                movies.append(pmf.w_Item[movie,:])
            recommended_movies_properties[inv] = movies
    #print(pred.get(0))
    print("ok")
    new_pred = {}
    for i in range(len(recommended_movies_properties)): 
        if(pred.get(i) is not None): # some users don't exist in the validation data
            mmr = MMR(recommended_movies_properties.get(i),pred.get(i),pmf.w_User[i,:],1,10) 
            new_pred[i] = mmr.rank()
            print(pred.get(i))
            print(new_pred.get(i))
    """
    print(new_pred.get(1))
    print(pred.get(1))
    print(new_pred[1][0])
    print(pred[1][0])
    """
    """
    print(mmr.compute_similarity(new_pred[1][0],pmf.w_User[1,:]))
    print(mmr.compute_similarity(new_pred[1][1],pmf.w_User[1,:]))
    print(mmr.compute_similarity(new_pred[1][2],pmf.w_User[1,:]))
    print(mmr.compute_similarity(new_pred[1][3],pmf.w_User[1,:]))

    print(mmr.compute_similarity(recommended_movies_properties[1][0],pmf.w_User[1,:]))
    print(mmr.compute_similarity(pred[1][1],pmf.w_User[1,:]))
    print(mmr.compute_similarity(pred[1][2],pmf.w_User[1,:]))
    print(mmr.compute_similarity(pred[1][3],pmf.w_User[1,:]))
    print(mmr.compute_similarity(pred[1][4],pmf.w_User[1,:]))

    print(new_pred.get(1))
    print(pred.get(1))
    print(new_pred[1][0])
    print(pred[1][0])
    """
    """
    intersection_cnt = {}

    for i in range(validation_data.shape[0]):
        if new_pred.get(int(validation_data[int(i), 0])) is not None: #if(int(validation_data[int(i), 0]) < len(new_pred)):
            if (int(validation_data[int(i), 1])) in (new_pred[int(validation_data[int(i), 0])]): # for each rating if the movie is in the movie list of predictions for the user
                intersection_cnt[validation_data[int(i), 0]] = intersection_cnt.get(validation_data[int(i), 0], 0) + 1 # we increment, second argument of get is default value
    invPairs_cnt = np.bincount(np.array(validation_data[:, 0], dtype='int32')) # counts the number of ratings for each user

 
    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
    print("precision_acc = " +str(precision_acc / len(inv_lst))+" recall_acc ="+str(recall_acc / len(inv_lst)))
    """
    """
    new_pred1 = {}
    for i in range(2):#len(recommended_movies)): 
        if(pred.get(i) is not None): # some users don't exist in the validation data
            mmr = MMR(recommended_movies.get(i),pred.get(i),pmf.w_User[i,:],0.9,10,pmf.mean_inv) 
            new_pred1[i] = mmr.rank()

    intersection_cnt = {}

    for i in range(validation_data.shape[0]):
        if new_pred1.get(int(validation_data[int(i), 0])) is not None: #if(int(validation_data[int(i), 0]) < len(new_pred)):
            if (int(validation_data[int(i), 1])) in (new_pred1[int(validation_data[int(i), 0])]): # for each rating if the movie is in the movie list of predictions for the user
                intersection_cnt[validation_data[int(i), 0]] = intersection_cnt.get(validation_data[int(i), 0], 0) + 1 # we increment, second argument of get is default value
    invPairs_cnt1 = np.bincount(np.array(validation_data[:, 0], dtype='int32')) # counts the number of ratings for each user

 
    precision_acc = 0.0
    recall_acc = 0.0
    for inv in inv_lst:
        precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
        recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt1[int(inv)])
    print("precision_acc = " +str(precision_acc / len(inv_lst))+" recall_acc ="+str(recall_acc /len(inv_lst)))
    """
    """
    print("precision_acc,recall_acc:" + str(pmf.topK(validation_data)))  
    """
    """
    print(recommended_movies[1])
    print(mmr.compute_similarity(recommended_movies[1][0],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][1],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][2],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][3],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][4],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][5],pmf.w_User[1,:]))
    print(mmr.compute_similarity(recommended_movies[1][6],pmf.w_User[1,:]))

    print(np.dot(recommended_movies[1][0], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][1], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][2], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][3], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][4], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][5], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][6], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][7], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    print(np.dot(recommended_movies[1][8], pmf.w_User[1, :]) + pmf.mean_inv)  # numpy.dot 点乘
    """
    """
    mmr_pred = {}
    for inv in inv_lst:
        mmr = MMR()
    """



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
    print("precision_acc,recall_acc:" + str(pmf.topK(validation_data)))
"""