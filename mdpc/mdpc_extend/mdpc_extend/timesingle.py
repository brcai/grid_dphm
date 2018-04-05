from density_peak_clustering import dp
from hausdorff_hierarchical import huasdorffHier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile, loadData
from sklearn import metrics
from dp_orig import dpOrig
from dphm import dphm
from ddbscan import ddbscan
import timeit
import numpy as np

def scalarTmp(data):
	for i in range(len(data[0])):
		maxVal = -10000.
		dataTmp = data.copy()
		for j in range(len(data)):
			if data[j][i] > maxVal: maxVal = data[j][i]
		for j in range(len(data)):
			if maxVal != 0: dataTmp[j][i] = data[j][i]/maxVal
	return dataTmp

if __name__ == '__main__':
	data = []
	label = []
	num = 0
	print('name of the dataset: ')
	dataSet = input()
	dataTmp, label = loadData(dataSet)
	print(dataSet)
	tmp = set(label)
	num = len(tmp)
	data = scalarTmp(dataTmp)
	#print('Enter dc: ')
	#dc = float(input())
	#print('enter lasso: ')
	#lasso = float(input())
	dc = 0.3
	lasso = 0.1
	inst = dpOrig()
	start = timeit.default_timer()
	dlabel, valid, dcenters = inst.eval(dc, num, 'nl', data)
	stop = timeit.default_timer()
	print('dpc takes' + str(stop-start))

	print('running density peak merge clustering......')
	inst = dphm()
	start = timeit.default_timer()
	inst.eval(dc, lasso, 'nl', data)
	stop = timeit.default_timer()
	print('dpmc takes' + str(stop-start))
	print('running density peak hausdorff clustering......')
	inst = huasdorffHier()
	start = timeit.default_timer()
	arsTmp, amiTmp, centers, clusters = inst.eval(dc, 'nl', data, label, 1000000)
	stop = timeit.default_timer()
	print('dphc takes' + str(stop-start))
	print('running dbscan clustering......')
	inst = ddbscan()
	start = timeit.default_timer()
	inst.eval(0.1, 2, 0, data, True)
	#eps, minSp, lasso, dataVecs, ifNormal
	stop = timeit.default_timer()
	print('dbscan takes' + str(stop-start))

	print("End of Test!")

