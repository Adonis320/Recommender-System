import math
from math import sqrt
from decimal import Decimal

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
