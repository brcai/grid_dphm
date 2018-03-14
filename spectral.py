from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import stopWords as stpw
import re
from sklearn.cluster import DBSCAN
from read_data import readDataFile
from sklearn import metrics
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


def read(dir):
	fp = open('C:/study/clustering dataset/' + dir)
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


if __name__ == "__main__":
	'''
	dx, dy, id, num = read('jain.txt')
	dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
	db = AffinityPropagation(damping=0.9, preference=-100).fit(dataVecs)
	dlabel = db.labels_
	colors = cm.rainbow(np.linspace(0, 1, 10 + 1))
	dpcolorMap2 = [dlabel[itm] for itm in range(len(dataVecs))]
	dx = [dataVecs[i][0] for i in range(len(dataVecs))]
	dy = [dataVecs[i][1] for i in range(len(dataVecs))]
	newMap = []
	for idx,itm in enumerate(dlabel):
		if itm == -1: newMap.append([0,0,0])
		else: newMap.append(dpcolorMap2[idx])

	plt.scatter(dx, dy, c=newMap, marker='.', s=100)
	#plt.xlabel('X')
	plt.ylabel('Y')
	plt.xticks([])
	plt.yticks([])
	plt.show()
	'''

	datasets = [['D31.txt', 31, 'gs'], ['Compound.txt', 5, 'nl'],['R15.txt', 15, 'gs'],['pathbased.txt', 3, 'nl'],['jain.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],['flame.txt', 2, 'nl'],
			 ['spiral.txt', 3, 'nl']]
	for dataset in datasets:
		print('running spectral......')
		print(dataset[0])
		dx, dy, id, num = read(dataset[0])
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		epss = 0.1
		minp = 0.0
		ars = 0.0
		
		db = spectral_clustering(affinity='nearest_neighbors',n_clusters=dataset[1]).fit(np.array(dataVecs))
		dlabel = db.labels_
		tmp = 10
		dd = []
		for itm in dlabel:
			if itm == -1: dd.append(tmp); tmp += 1
			else: dd.append(itm)
		arsTmp = metrics.adjusted_mutual_info_score(id, dd)
		if arsTmp > ars:
			ars = arsTmp
		print('arsdpc = '+str(ars)+'\n')
		#print('bestparam: eps='+str(epss)+'  minp='+str(minp)+'\n')
	'''
	while True:
		print('eps:')
		i = float(input())
		print('minSp:')
		j = float(input())
		print('lasso:')
		k = float(input())
		inst.run('Aggregation.txt', i, j, k, 'nl')
	dx, dy, id, num = inst.read('pathbased.txt')
	dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
	while True:
		print('eps:')
		i = float(input())
		print('minSp:')
		j = float(input())
		dbs = DBSCAN(eps=i, min_samples=j).fit(dataVecs)
		dlabel = dbs.labels_
		dpcolorMap1 = ['k' for i in range(len(dx))]
		plt.scatter(dx, dy, c=dpcolorMap1)
		for i in range(len(dx)):
			plt.annotate(dlabel[i], (dx[i],dy[i]))
		plt.show()
	'''
	'''
	#done normal distance
	clusters = inst.run('Compound.txt', 0.06, 2, 'nl')
	clusters = inst.run('pathbased.txt', 0.085, 0.8, 'nl')
	clusters = inst.run('Aggregation.txt', 0.036, 1, 'nl')
	clusters = inst.run('spiral.txt', 0.12, 1, 'nl')
	inst.run('flame.txt', 0.1, 1, 'nl')
	inst.run('a3.txt', 0.020, 2, 'nl')

	#done Gaussian distance
	clusters = inst.run('R15.txt', 0.040, 1, 'gs')
	inst.run('s1.txt', 0.028, 1, 'gs')
	clusters = inst.run('D31.txt', 0.021, 1, 'gs')
	clusters = inst.run('s4.txt', 0.06, 7, 'gs')
	'''
	print('The end...')


