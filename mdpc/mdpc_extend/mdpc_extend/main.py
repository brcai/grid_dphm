from density_peak_clustering import dp
from dpmc_hausdorff import dph
from ddbscan import ddbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile, loadData
from sklearn import metrics
from dp_orig_bk import dpOrig
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump



if __name__ == '__main__':
	label = []
	num = 0
	print('name of the dataset: ')
	dataSet = input()
	data, label = loadData(dataSet)
	tmp = set(label)
	num = len(tmp)

	print('running density peak clustering......')
	dc = 0.1
	amidpc = 0.0
	bestdc = 0.1
	inst = dpOrig()
	for i in frange(0.1, 1, 0.1):
		dlabel, valid, centers = inst.eval(i, num, 'nl', data)
		if valid:
			arsTmp = metrics.adjusted_rand_score(label, dlabel)
			amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
			if amiTmp > amidpc:
				amidpc = amiTmp
	print('amidpc = '+str(amidpc)+'\n')

	print('running density peak merge clustering......')
	dmc = 0.1
	lasso = 0.1
	amidpmc = 0.0
	inst = dp()
	cnt = 0
	for i in frange(0.1, 1, 0.1):
		for j in frange(0.1, 2, 0.1):
			dlabel, valid, centers = inst.eval(i, j, 'nl', data)
			if valid:
				amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
				if amiTmp > amidpmc:
					amidpmc = amiTmp
			else: break
	print('amidpmc = '+str(amidpmc)+'\n')

	print('running dbscan......')
	amidbscan = 0.0
	inst = ddbscan()
	for i in frange(0.1, 5, 0.1):
		for j in range(1, 15, 1):
			dbs = DBSCAN(eps=i, min_samples=j).fit(data)
			dlabel = dbs.labels_
			amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
			if amiTmp > amidbscan:
				amidbscan = amiTmp
			else: break
	print('amidbscan = '+str(amidbscan)+'\n')


	print("End of Test!")
