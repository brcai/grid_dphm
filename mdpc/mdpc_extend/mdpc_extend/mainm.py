from density_peak_clustering import dp
from ddbscan import ddbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile, loadData
from sklearn import metrics
from dp_orig import dpOrig

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def scalarTmp(data):
	for i in range(len(data[0])):
		maxVal = -10000.
		dataTmp = data.copy()
		for j in range(len(data)):
			if data[j][i] > maxVal: maxVal = data[j][i]
		for j in range(len(data)):
			if maxVal != 0: dataTmp[j][i] = data[j][i]/maxVal
	return dataTmp

def calcAcc(tlabel, clabel):
	acc = 0.
	tidx = set(tlabel)
	cidx = set(clabel)
	dataLen = len(tlabel)
	llabel = tlabel if len(tlabel) >= len(clabel) else clabel
	slabel = tlabel if len(tlabel) < len(clabel) else tlabel

	return acc


if __name__ == '__main__':
	data = []
	label = []
	num = 0
	print('name of the dataset: ')
	dataSet = input()
	fp = open(dataSet+'-dpc.txt', 'w')
	dataTmp, label = loadData(dataSet)
	tmp = set(label)
	num = len(tmp)

	print('running density peak clustering......')
	fp.write('running density peak clustering......')
	dc = 0.1
	arsdpc = 0.0
	amidpc = 0.0
	bestdc = 0.1
	inst = dpOrig()
	for i in frange(0.1, 1, 0.1):
		dlabel, valid = inst.eval(i, num, 'nl', dataTmp)
		if valid:
			arsTmp = metrics.adjusted_rand_score(label, dlabel)
			amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
			if arsTmp > arsdpc and amiTmp > amidpc:
				arsdpc = arsTmp
				amidpc = amiTmp
				bestdc = i
	print('arsdpc = '+str(arsdpc)+'\n')
	print('amidpc = '+str(amidpc)+'\n')
	print('bestdc = '+str(bestdc)+'\n')
	fp.write('arsdpc = '+str(arsdpc)+'\n')
	fp.write('amidpc = '+str(amidpc)+'\n')
	fp.write('bestdc = '+str(bestdc)+'\n')

	print('running density peak merge clustering......')
	fp.write('running density peak merge clustering......')
	dmc = 0.1
	lasso = 0.1
	arsdpmc = 0.0
	amidpmc = 0.0
	bestdmc = 0.1
	bestlasso = 0.1
	inst = dp()
	for i in frange(0.1, 1, 0.1):
		for j in frange(0.1, 2, 0.1):
			dlabel, valid = inst.eval(i, j, 'nl', dataTmp)
			if valid:
				arsTmp = metrics.adjusted_rand_score(label, dlabel)
				amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
				if arsTmp > arsdpmc and amiTmp > amidpmc:
					arsdpmc = arsTmp
					amidpmc = amiTmp
					bestdmc = i
					bestlasso = j
			
					print(bestdmc)
					print(bestlasso)
					print(arsdpmc)
					print(amidpmc)
			else: break
	print('arsdpmc = '+str(arsdpmc)+'\n')
	print('amidpmc = '+str(amidpmc)+'\n')
	print('bestdmc = '+str(bestdmc)+'\n')
	print('bestlasso = '+str(bestlasso)+'\n')
	fp.write('arsdpmc = '+str(arsdpmc)+'\n')
	fp.write('amidpmc = '+str(amidpmc)+'\n')
	fp.write('bestdmc = '+str(bestdmc)+'\n')
	fp.write('bestlasso = '+str(bestlasso)+'\n')

	print("End of Test!")
	fp.write("End of Test!")
	fp.close()
