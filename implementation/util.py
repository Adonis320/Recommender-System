import math
import numpy as np

class UTIL(object):
    def __init__(self):
        pass

    def compute_similarity(self,v1,v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return round((prod / (len1 * len2)))

    def dot_product(self,v1, v2):
        return sum([x*y for x,y in zip(v1,v2)])

    def evaluateMMR(self,pred,test_data,k,inv_lst):
        intersection_cnt = {}
        for i in range(test_data.shape[0]):
            if pred.get(int(test_data[int(i), 0])) is not None:
                if (int(test_data[int(i), 1])) in (pred[int(test_data[int(i), 0])]): # for each rating if the movie is in the movie list of predictions for the user
                    intersection_cnt[test_data[int(i), 0]] = intersection_cnt.get(test_data[int(i), 0], 0) + 1 # we increment, second argument of get is default value
        invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32')) # counts the number of ratings for each user
        
        
        precision_acc = 0.0
        recall_acc = 0.0
        f1 = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
        f1 = 2*(precision_acc/len(inv_lst))*(recall_acc/len(inv_lst))/((precision_acc/len(inv_lst))+(recall_acc/len(inv_lst)))
        print("MMR: precision_acc = " +str(precision_acc / len(inv_lst))+" recall_acc = "+str(recall_acc / len(inv_lst)) + " F1 = "+ str(f1))

