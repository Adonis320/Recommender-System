import numpy as np
import math
from scipy import spatial

class MMR(object):
    def __init__(self, item_vector, recommended_movies, user_vector, _lambda, k=10):
        
        self.item_vector = item_vector  # Vector of movies with caracteristics
        self.recommended_movies = recommended_movies # Ranked movies
        self.user_vector = user_vector
        self._lambda = _lambda  # Parameter for MMR    

        self.indexed_movies = {} # movies with rank
        self.similarity_matrix = []
        self.similarity_with_user = []
        self.k = k
        
        a = 0
        for movie in recommended_movies:
            self.indexed_movies[movie]= a    
            a+=1 

        #self.compute_similarity_matrix_default()
        self.compute_similarity_matrix()
        #self.compute_similarity_with_user_default()
        self.compute_similarity_with_user()

        print("DONE")
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
        r = self.recommended_movies
        r_s = r # R\S
        r_s = np.array(r_s) # to use np.delete

        while(len(s)<self.k):#len(r)): # we rerank all the movies
            scores_1 = []
            index_scores_1 = []
            for i in r_s:
                scores_2 = []
                for j in s:
                    scores_2.append(self.similarity_matrix[self.indexed_movies.get(i)][self.indexed_movies.get(j)])
                if(len(scores_2)==0):
                    max_value_2 = 0
                else:
                    max_value_2 = max(scores_2)
                scores_1.append(self._lambda*self.similarity_with_user[self.indexed_movies.get(i)]-(1-self._lambda)*max_value_2)
                index_scores_1.append(i)  
            max_value_1 = max(scores_1)
            max_index_1 = scores_1.index(max_value_1)
            movie = index_scores_1[max_index_1]
            s.append(movie) 
            index_delete = np.argwhere(r_s==movie)
            r_s = np.delete(r_s, index_delete)
        return s

    def compute_similarity_matrix(self):
        for m1 in self.recommended_movies:
            sim = []
            for m2 in self.recommended_movies:
               sim.append(self.compute_similarity(self.item_vector[self.indexed_movies.get(m1)],self.item_vector[self.indexed_movies.get(m2)]))
            self.similarity_matrix.append(sim)            

    def compute_similarity(self,v1,v2):

        #return  spatial.distance.cosine(v1, v2)
        
        prod = self.dot_product(v1, v2)
        len1 = math.sqrt(self.dot_product(v1, v1))
        len2 = math.sqrt(self.dot_product(v2, v2))
        return round((prod / (len1 * len2)),3)
        

    def dot_product(self,v1, v2):
        return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

    def compute_similarity_with_user(self):
        for m in self.recommended_movies:
            self.similarity_with_user.append(self.compute_similarity(self.item_vector[self.indexed_movies.get(m)],self.user_vector))
