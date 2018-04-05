from density_peak_clustering import dp
from ddbscan import ddbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile
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

def loadData(dataSet):
	data = []
	label = []
	if dataSet == 'iris':
		data, label = readDataFile.iris()
	elif dataSet == 'wine':
		data, label = readDataFile.wine()
	elif dataSet == 'heart':
		data, label = readDataFile.heart()
	elif dataSet == 'wdbc':
		data, label = readDataFile.wdbc()
	elif dataSet == 'ionosphere':
		data, label = readDataFile.ionoshpere()
	elif dataSet == 'waveform':
		data, label = readDataFile.waveform()
	elif dataSet == 'pendigits':
		data, label = readDataFile.pendigits()
	elif dataSet == 'japanesev':
		data, label = readDataFile.japanesev()
	elif dataSet == 'monk3':
		data, label = readDataFile.monk3()
	elif dataSet == 'movement':
		data, label = readDataFile.movement_libras()
	elif dataSet == 'semeion':
		data, label = readDataFile.semeion()
	elif dataSet == 'spambase':
		data, label = readDataFile.spambase()
	elif dataSet == 'wholesale':
		data, label = readDataFile.wholesale()
	elif dataSet == 'zoo':
		data, label = readDataFile.zoo()
	elif dataSet == 'spect':
		data, label = readDataFile.spect()
	elif dataSet == 'pima':
		data, label = readDataFile.pima_india_diabetes()
	elif dataSet == 'seeds':
		data, label = readDataFile.seeds()
	elif dataSet == 'bcw':
		data, label = readDataFile.bcw()
	elif dataSet == 'sonar':
		data, label = readDataFile.sonar()
	elif dataSet == 'pageb':
		data, label = readDataFile.pageb()
	elif dataSet == 'bupa':
		data, label = readDataFile.bupa()
	elif dataSet == 'ecoli':
		data, label = readDataFile.ecoli()
	elif dataSet == 'musk':
		data, label = readDataFile.musk()

	dataTmp = scalarTmp(data)
	return dataTmp, label

if __name__ == '__main__':
	data = []
	label = []
	num = 0
	print('name of the dataset: ')
	dataSet = input()
	fp = open(dataSet+'-dbscan.txt', 'w')
	dataTmp, label = loadData(dataSet)
	tmp = set(label)
	num = len(tmp)

	print('running dbscan......')
	fp.write('running dbscan......')
	db1 = 0.1
	db2 = 2
	arsdbscan = 0.0
	amidbscan = 0.0
	besteps = 0.
	bestminSp = 0
	bestlasso = 0.
	inst = ddbscan()
	for i in frange(0.1, 1, 0.1):
		for j in range(1, 15, 1):
			for k in frange(0.1, 0.2, 0.1):
				dlabel, valid = inst.eval(i, j, k, dataTmp, True)
				if valid:
					arsTmp = metrics.adjusted_rand_score(label, dlabel)
					amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
					if arsTmp > arsdbscan and amiTmp > amidbscan:
						arsdbscan = arsTmp
						amidbscan = amiTmp
						besteps = i
						bestminSp = j
						bestlasso = k
				else: break
	print('arsdbscan = '+str(arsdbscan)+'\n')
	print('amidbscan = '+str(amidbscan)+'\n')
	print('besteps = '+str(besteps)+'\n')
	print('bestminSp = '+str(bestminSp)+'\n')
	print('bestlasso = '+str(bestlasso)+'\n')
	fp.write('arsdbscan = '+str(arsdbscan)+'\n')
	fp.write('amidbscan = '+str(amidbscan)+'\n')
	fp.write('besteps = '+str(besteps)+'\n')
	fp.write('bestminSp = '+str(bestminSp)+'\n')
	fp.write('bestlasso = '+str(bestlasso)+'\n')


	print('running dbscan with merging......')
	fp.write('running dbscan with merging......')
	db1 = 0.1
	db2 = 2
	arsdbscan = 0.0
	amidbscan = 0.0
	besteps = 0.
	bestminSp = 0
	bestlasso = 0.
	inst = ddbscan()
	for i in frange(0.1, 1, 0.1):
		for j in range(1, 15, 1):
			for k in frange(-1, 1, 0.1):
				dlabel, valid = inst.eval(i, j, k, dataTmp, False)
				if valid:
					arsTmp = metrics.adjusted_rand_score(label, dlabel)
					amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
					if arsTmp > arsdbscan and amiTmp > amidbscan:
						arsdbscan = arsTmp
						amidbscan = amiTmp
						besteps = i
						bestminSp = j
						bestlasso = k
				else: break
	print('arsdbscan = '+str(arsdbscan)+'\n')
	print('amidbscan = '+str(amidbscan)+'\n')
	print('besteps = '+str(besteps)+'\n')
	print('bestminSp = '+str(bestminSp)+'\n')
	print('bestlasso = '+str(bestlasso)+'\n')
	fp.write('arsdbscan = '+str(arsdbscan)+'\n')
	fp.write('amidbscan = '+str(amidbscan)+'\n')
	fp.write('besteps = '+str(besteps)+'\n')
	fp.write('bestminSp = '+str(bestminSp)+'\n')
	fp.write('bestlasso = '+str(bestlasso)+'\n')

	'''
	print('running dbscan......')
	db1 = 0.1
	db2 = 2
	arsdbscan = 0.0
	amidbscan = 0.0
	bestEps = 0.1
	bestMinpts = 1
	for i in frange(0.1, 10, 0.1):
		for j in range(1, 20, 1):
			dbs = DBSCAN(eps=i, min_samples=j).fit(dataTmp)
			dlabel = dbs.labels_
			arsTmp = metrics.adjusted_rand_score(label, dlabel)
			amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
			if arsTmp > arsdbscan and amiTmp > amidbscan:
				arsdbscan = arsTmp
				amidbscan = amiTmp
				bestEps = i
				bestMinpts = j
	print('arsdbscan = '+str(arsdbscan)+'\n')
	print('amidbscan = '+str(amidbscan)+'\n')
	print('bestEps = '+str(bestEps)+'\n')
	print('bestMinpts = '+str(bestMinpts)+'\n')
	'''
	'''	
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
		for j in frange(-2, 0, 0.1):
			dlabel, valid = inst.eval(i, j, 'gs', dataTmp)
			if valid:
				arsTmp = metrics.adjusted_rand_score(label, dlabel)
				amiTmp = metrics.adjusted_mutual_info_score(label, dlabel)
				if arsTmp > arsdpmc and amiTmp > amidpmc:
					arsdpmc = arsTmp
					amidpmc = amiTmp
					bestdmc = i
					bestlasso = j
			else: break
	print('arsdpmc = '+str(arsdpmc)+'\n')
	print('amidpmc = '+str(amidpmc)+'\n')
	print('bestdmc = '+str(bestdmc)+'\n')
	print('bestlasso = '+str(bestlasso)+'\n')
	fp.write('arsdpmc = '+str(arsdpmc)+'\n')
	fp.write('amidpmc = '+str(amidpmc)+'\n')
	fp.write('bestdmc = '+str(bestdmc)+'\n')
	fp.write('bestlasso = '+str(bestlasso)+'\n')
	'''
	print("End of Test!")
	fp.write("End of Test!")
	fp.close()
