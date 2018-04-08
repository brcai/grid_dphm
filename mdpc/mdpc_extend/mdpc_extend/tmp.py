import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics
from func import *

def calcMaxDist(dir):
	dx, dy, id, num = read(dir)
	dataVecs = [[dx[i],dy[i]] for i in range(len(dx))]
	rawDists = set()
	for i in range(len(dataVecs)):
		for j in range(i + 1, len(dataVecs)):
			dist = euclidean(dataVecs[i], dataVecs[j])
			rawDists.add(dist)
	dists = list(rawDists)
	dists.sort()
	return dists


#read dataset
def read(dir):
	fp = open('C:/study/datasets/synthetic/' + dir)
	dx = []
	dy = []
	id = []
	num = 0
	clusters = []
	for line in fp.readlines():
		raw = re.split('[ |\t]', line)
		tmp = [itm for itm in raw if itm != '']
		arr = [float(itm.replace('\n', '')) for itm in tmp]
		dx.append(arr[0])
		dy.append(arr[1])
		if len(arr) == 3: 
			id.append(int(arr[2]))
			if arr[2] not in clusters:
				clusters.append(arr[2])
				num += 1
	return dx, dy, id, num

if __name__ == '__main__':
	print('Calculating max distances:')
	print('pathbased: '+str(calcMaxDist('pathbased.txt')[-1]))
	print('Compound: '+str(calcMaxDist('Compound.txt')[-1]))
	print('jain: '+str(calcMaxDist('jain.txt')[-1]))
	print('flame: '+str(calcMaxDist('flame.txt')[-1]))
	print('Aggregation: '+str(calcMaxDist('Aggregation.txt')[-1]))
	print('spiral: '+str(calcMaxDist('spiral.txt')[-1]))
	print('R15: '+str(calcMaxDist('R15.txt')[-1]))
	print('D31: '+str(calcMaxDist('D31.txt')[-1]))

	print("End of Test!")