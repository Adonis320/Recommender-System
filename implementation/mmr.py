import numpy as np
import math
from scipy import spatial

class MMR(object):
    def __init__(self, recommended_movies, userId, similarity_matrix, similarity_with_user, _lambda, k=10):
        
        self.recommended_movies = recommended_movies # Ranked movies
        self.userId = userId
        self._lambda = _lambda  # Parameter for MMR    

        self.similarity_matrix = similarity_matrix # similarity matrix between the movies themselves
        self.similarity_with_user = similarity_with_user # similarity between each movie and the user
        self.k = k # number of movies to return
        """
        For testing purposes: MMR should return [102,302,4,70,51]
        self._lambda = 0.5
        self.similarity_with_user = [0.91, 0.9, 0.5, 0.06, 0.63, 1]
        self.similarity_matrix = [[1, 0.11, 0.23, 0.76, 0.25],[0.11,1,0.29,0.57,0.51],[0.23,0.29,1,0.02,0.2],[0.76,0.57,0.02,1,0.33],[0.25,0.51,0.2,0.33,1]]
        self.indexed_movies = {102:0,302:1,4:2,51:3,70:4}
        self.recommended_movies = [102,302,4,51,70]
        """
    
    def rank(self):
        s = []
        r_s = self.recommended_movies # R\S
        r_s = np.array(r_s) # to use np.delete

        while(len(s)<self.k): # we rerank all the movies
            scores_1 = [] # lamba*sim(Di,Q)-(1-lambda)*max(sim(Di,Dj))
            index_scores_1 = [] # to reindex the movies while reiterating

            for i in r_s:
                scores_2 = [] # sim(Di,Dj)
                for j in s:
                    scores_2.append(self.similarity_matrix[i][j])
                if(len(scores_2)==0):# for the first iteration
                    max_value_2 = 0
                else:
                    max_value_2 = max(scores_2) # max(sim(Di,Dj))
                scores_1.append(self._lambda*self.similarity_with_user.get(self.userId)[i]-(1-self._lambda)*max_value_2)
                index_scores_1.append(i)  

            max_value_1 = max(scores_1) # maximal MMR
            max_index_1 = scores_1.index(max_value_1) # find the index of the movie with maximal MMR
            movie = index_scores_1[max_index_1] # find the movie with maximal MMR
            s.append(movie)

            index_delete = np.argwhere(r_s==movie) # index of the movie to delete in R\S
            r_s = np.delete(r_s, index_delete)
        return s
