import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.cluster import DBSCAN
from sklearn import metrics

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def euclidean(vec1, vec2):
	dist = 0.
	for i in range(len(vec1)):
		dist += pow((vec1[i] - vec2[i]), 2)
	res = np.sqrt(dist)
	return res
