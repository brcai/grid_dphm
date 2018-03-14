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


class ddbscan:
	def calcP(self, distMat, eps):
		pRank = []
		data2P = [0. for i in range(len(distMat))]
		for idx, itm in enumerate(distMat):
			cnt = 0.
			for j in itm:
				if j <= eps: cnt += 1;
				#if j <= eps: cnt +=np.exp(-j**2/eps**2)
			pRank.append([cnt, idx])
			data2P[idx] = cnt
		pRank.sort(reverse=True)
		return pRank, data2P

	def assignCluster(self, dataVecs, epsRaw, minSp, lasso, ifNormal):
		visited = {i: False for i in range(len(dataVecs))}
		centers = []
		currentC = -1
		currentD = 0.
		distMat, maxDist = self.calcDists(dataVecs)
		eps = epsRaw*maxDist
		pRank, data2P = self.calcP(distMat, eps)
		clusterId = 0
		clusters = []
		data2Cluster = [-1 for i in range(len(dataVecs))]

		for data in pRank:
			idx = data[1] 
			if visited[idx] == True: continue
			neighbour = self.getEpsNeighbour(idx, distMat, eps)
			if len(neighbour) < minSp: continue
			visited[idx] = True
			center = idx
			centerD = 0
			baseD = data[0]
			coreList = [idx]
			clusters.append([idx])
			data2Cluster[idx] = clusterId
			while len(neighbour) != 0:
				newNeighbour = []
				for core in neighbour:
					if visited[core] == True: continue
					clusters[clusterId].append(core)
					data2Cluster[core] = clusterId
					visited[core] = True
					coreNeighbour = self.getEpsNeighbour(core, distMat, eps)
					coreDense = len(coreNeighbour)
					if coreDense >= minSp:
						if coreDense > centerD: centerD = coreDense; center = core
						adjustedNeighbour = []
						if ifNormal: adjustedNeighbour = self.getEpsNeighbour(core, distMat, eps)
						else: adjustedNeighbour = self.getEpsNeighbour(core, distMat, eps*(data2P[core]/baseD)*lasso)
						for itm in adjustedNeighbour:
							if itm not in newNeighbour and visited[itm] != True: newNeighbour.append(itm)
				neighbour = newNeighbour
				test = [idx for idx in visited if visited[idx] == True]
				#self.plotCluster(data2Cluster, [data[0] for data in dataVecs], [data[1] for data in dataVecs], test)
			clusterId += 1
			centers.append(center)

		return data2Cluster, centers

	def getEpsNeighbour(self, i, distMat, eps):
		neighbour = []
		for idx, itm in enumerate(distMat[i]):
			if idx != i and itm <= eps: neighbour.append(idx)
		return neighbour

	def calcDists(self, dataVecs):
		distMat = []
		tmpSet = set()
		for i in range(len(dataVecs)):
			tmp = []
			for j in range(len(dataVecs)):
				dist = stpw.euclidean(dataVecs[i], dataVecs[j])
				tmp.append(dist)
				tmpSet.add(dist)
			distMat.append(tmp)
		rankedDist = list(tmpSet)
		rankedDist.sort()
		return distMat, rankedDist[-1]

	#read dataset
	def read(self, dir):
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

	def plotCluster(self, data2Cluster, dx, dy, centers):
		#colors = cm.rainbow(np.linspace(0, 1, 20))
		dpcolorMap1 = ['k' for i in range(len(dx))]
		for itm in centers:
			dpcolorMap1[itm] = 'r'
		plt.scatter(dx, dy, c=dpcolorMap1)
		for i in range(len(dx)):
			plt.annotate(data2Cluster[i], (dx[i],dy[i]))
		plt.show()
		return

	#density peak clustering flow
	def run(self, dir, eps, minSp, lasso, kl):
		dx, dy, id, num = self.read(dir)
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		data2Cluster, centers = self.assignCluster(dataVecs, eps, minSp, lasso, False)
		#self.plotCluster(data2Cluster, dx, dy, centers)
		label = [data2Cluster[i] for i in data2Cluster]
		return data2Cluster, label

	#for evaluation
	def eval(self, eps, minSp, lasso, dataVecs, ifNormal):
		data2Cluster, centers = self.assignCluster(dataVecs, eps, minSp, lasso, ifNormal)
		#self.plotClusterRes(wordCluster, dataVecs, centers, finalRes)
		label = [data2Cluster[i] for i in range(len(data2Cluster))]
		return label, True

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
	datasets = [['Compound.txt', 5, 'nl'],['pathbased.txt', 3, 'nl'],['jain.txt', 2, 'nl']]
	'''
	datasets = [['Compound.txt', 5, 'nl'],['pathbased.txt', 3, 'nl'],['jain.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],['D31.txt', 31, 'gs'], ['flame.txt', 2, 'nl'],
			 ['spiral.txt', 3, 'nl'],['R15.txt', 15, 'gs']]
	for dataset in datasets:
		print('running dbscan......')
		print(dataset[0])
		dx, dy, id, num = read(dataset[0])
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		epss = 0.1
		minp = 0.0
		ars = 0.0
		for i in frange(0.1, 5, 0.1):
			for j in range(1, 20, 1):
				db = DBSCAN(eps=i, min_samples=j).fit(dataVecs)
				dlabel = db.labels_
				tmp = 10
				dd = []
				for itm in dlabel:
					if itm == -1: dd.append(tmp); tmp += 1
					else: dd.append(itm)
				arsTmp = metrics.adjusted_mutual_info_score(id, dd)
				if arsTmp > ars:
					ars = arsTmp
					epss = i
					minp = j
		print(epss,minp)
		print('arsdpc = '+str(ars)+'\n')
		#print('bestparam: eps='+str(epss)+'  minp='+str(minp)+'\n')
	'''
	dir = 'jain.txt'
	dx, dy, id, num = read(dir)
	dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
	dbs = DBSCAN(eps=2.5, min_samples=1).fit(dataVecs)
	dlabel = dbs.labels_
	tmp = 10
	dd = []
	for itm in dlabel:
		if itm == -1: dd.append(tmp); tmp += 1
		else: dd.append(itm)
	print('arsdpc = '+str(metrics.adjusted_mutual_info_score(id, dd))+'\n')
	colors = cm.rainbow(np.linspace(0, 1, 7))
	dpcolorMap2 = [colors[dlabel[itm]] for itm in range(len(dataVecs))]

	dx = [dataVecs[i][0] for i in range(len(dataVecs))]
	dy = [dataVecs[i][1] for i in range(len(dataVecs))]
	plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=150, zorder=1, alpha=0.8, edgecolor='k')
	plt.text(13.5, 1.5, r'$NMI=$'+str(0.7549), fontsize=15, color='black')
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')
	plt.savefig("C:\\study\\8data\\"+dir.split('.')[0]+"_dbscan.png", bbox_inches='tight', pad_inches = 0)
	#plt.savefig("C:\\study\\8data\\"+dir.split('.')[0]+"_dpha.png")
	plt.show()
	#done normal distance
	#clusters = inst.run('Compound.txt', 2.5, 10, 'nl')
	#clusters = inst.run('pathbased.txt', 1.8, 1, 'nl')
	#clusters = inst.run('jain.txt', 2.5, 1, 'nl')
	#clusters = inst.run('Aggregation.txt', 0.036, 1, 'nl')
	#clusters = inst.run('spiral.txt', 0.12, 1, 'nl')
	#inst.run('flame.txt', 0.1, 1, 'nl')
	#inst.run('a3.txt', 0.020, 2, 'nl')

	#done Gaussian distance
	#clusters = inst.run('R15.txt', 0.040, 1, 'gs')
	#inst.run('s1.txt', 0.028, 1, 'gs')
	#clusters = inst.run('D31.txt', 0.021, 1, 'gs')
	#clusters = inst.run('s4.txt', 0.06, 7, 'gs')
	print('The end...')