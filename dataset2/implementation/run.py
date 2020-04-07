# copied and modified from https://github.com/louiseGAN514/Probabilistic-Matrix-Factorization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from probabilistic_matrix_factorization import PMF
from load_data import load_training_data, load_validation_data

if __name__ == "__main__":
    train_path = "../data/training/binarized/ratings.csv"
    validation_path = "../data/validation/binarized/ratings.csv"

    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
    train_data = load_training_data()
    validation_data = load_validation_data()
    #print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    #train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.fit(train_data, validation_data)

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