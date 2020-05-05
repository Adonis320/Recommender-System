import math
import numpy as np

class UTIL(object):
    def __init__(self):
        pass

    def compute_similarity(self,v1,v2):
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return abs((prod / (len1 * len2)))

    def dot_product(self,v1, v2):
        return sum([x*y for x,y in zip(v1,v2)])

    def reorder_relevance(self,pred,user_similarity,user,k):
        new_pred = []
        predictions = np.array(pred)

        for i in range(k):
            index_scores = []
            scores = []
            for movie in predictions:
                index_scores.append(movie)
                scores.append(user_similarity.get(user)[movie])
            max_value = max(scores)
            max_index = scores.index(max_value)
            new_pred.append(predictions[max_index])
            predictions = np.delete(predictions,max_index)

        return new_pred

    def evaluateMMR(self,pred,test_data,k,user_similarity,mmr_lambda):
        inv_lst = np.unique(test_data[:, 0]) # unique users
        mrr = 0
        positive_pred = {}
        ndcg = 0

        for inv in inv_lst: 
            positive_pred[inv] = []  

        intersection_cnt = {}
        counter = 0
        for i in range(test_data.shape[0]):
            if pred.get(int(test_data[int(i), 0])) is not None:
                if (int(test_data[int(i), 1])) in (pred[int(test_data[int(i), 0])]): # for each rating if the movie is in the movie list of predictions for the user
                    intersection_cnt[test_data[int(i), 0]] = intersection_cnt.get(test_data[int(i), 0], 0) + 1 # we increment, second argument of get is default value
                    positive_pred[test_data[i,0]].append(test_data[i,1])
        invPairs_cnt = np.bincount(np.array(test_data[:, 0], dtype='int32')) # counts the number of ratings for each user
        
        ndcg_mean_counter = 0
        ndcg = 0
        for inv in inv_lst: 
            ranked = False
            if(pred.get(inv) is not None):
                i = 0
                dcg = 0
                idcg = 0
                for movie in pred[inv]:
                    i+=1
                    dcg += user_similarity.get(inv)[movie]/math.log(i+1,2)
                    if(movie in positive_pred[inv]):
                        if(ranked == False):
                            ranked = True
                            mrr += 1/(np.where(pred[inv]==movie)[0][0]+1)
                i = 0
                for movie in self.reorder_relevance(pred[inv],user_similarity,inv,k):
                    i+=1
                    idcg += user_similarity.get(int(inv))[int(movie)]/math.log(i+1,2)
                ndcg += dcg/idcg
                ndcg_mean_counter += 1

        precision_acc = 0.0
        recall_acc = 0.0
        f1 = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k) # how many movies hit
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])
        f1 = 2*(precision_acc/len(inv_lst))*(recall_acc/len(inv_lst))/((precision_acc/len(inv_lst))+(recall_acc/len(inv_lst)))
        print("MMR (lambda = "+ str(mmr_lambda)+"): precision_acc = " +str(precision_acc / len(inv_lst))+" recall_acc = "+str(recall_acc / len(inv_lst)) + " F1 = "+ str(f1) + " MRR = " + str(mrr/len(inv_lst)) + " NDCG = " + str(ndcg/ndcg_mean_counter))

