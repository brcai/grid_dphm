from density_peak_clustering import dp
from dpmc_hausdorff import dph
from hausdorff_hierarchical import huasdorffHier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile, loadData
from sklearn import metrics
from dp_orig import dpOrig
from sklearn.decomposition import PCA

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
	print('name of the dataset: ')
	dataSet = input()
	data, label = loadData(dataSet)
	tmp = set(label)
	num = len(tmp)
	dataTmp = PCA(n_components=3).fit_transform(data)

	mdl = input('c or h or m?')
	if mdl == 'c':
		inst = dp()
		print('running density peak clustering......')
		print('enter dc: ')
		dc = float(input())
		arsdpc = 0.0
		amidpc = 0.0
		bestdc = 0.1
		dlabel, valid, dcenters = inst.eval(dc, num, 'nl', dataTmp)
		if valid:
			arsTmp = metrics.adjusted_rand_score(label, dlabel)
			amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
			if arsTmp > arsdpc and amiTmp > amidpc:
				arsdpc = arsTmp
				amidpc = amiTmp
				bestdc = dc
	
		print('arsdpmc = '+str(arsdpc)+'\n')
		print('amidpmc = '+str(amidpc)+'\n')
		print('bestdmc = '+str(bestdc)+'\n')
		print('number of clusters is '+str(len(set(dlabel)))+'\n')
		print(dcenters)

	elif mdl == 'h':
		inst = huasdorffHier()
		i = float(input('enter dc for Hausdorff: '))
		arsTmp, amiTmp, centers, clusterWords = inst.eval(i, 'nl', dataTmp, label)
		print('arsdpc = '+str(arsTmp)+'\n')
		print('amidpc = '+str(amiTmp)+'\n')

	else:
		print('running density peak merge clustering......')
		print('enter dmc: ')
		dmc = float(input())
		while True:
			print('enter lasso: ')
			lasso = float(input())
			arsdpmc = 0.0
			amidpmc = 0.0
			bestdmc = 0.1
			bestlasso = 0.1
			inst = dp()
			dmlabel, valid, dmcenters = inst.eval(dmc, lasso, 'nl', dataTmp)
			if valid:
				arsTmp = metrics.adjusted_rand_score(label, dmlabel)
				amiTmp = metrics.adjusted_mutual_info_score(label, dmlabel)
				if arsTmp > arsdpmc and amiTmp > amidpmc:
					arsdpmc = arsTmp
					amidpmc = amiTmp
					bestdmc = dmc
					bestlasso = lasso
	
			print('arsdpmc = '+str(arsdpmc)+'\n')
			print('amidpmc = '+str(amidpmc)+'\n')
			print('bestdmc = '+str(bestdmc)+'\n')
			print('bestlasso = '+str(bestlasso)+'\n')
			print('number of clusters is '+str(len(set(dmlabel)))+'\n')
			print(dmcenters)

	print("End of Test!")
