import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import stopWords as stpw
import re
from sklearn import metrics
from read_data import readDataFile, loadData
from sklearn.decomposition import PCA

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


class huasdorffHier:
	#calculate p and delta
	def calcParams(self, dataVecs, dc, kl):
		wordParam = {}
        #calculate p as param[0], neighbours as param[1]
		maxp = 0
		maxData = 0
		maxDist = 0.0
		for i in range(0, len(dataVecs)):
			cnt = 0
			neighbours = []
			for j in range(0, len(dataVecs)):
				if i!=j:
					tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
					tmpDist = 0.
					if tmp < dc: 
						#normal regularization
						if kl == 'nl':
							cnt += 1-(tmp**2/dc**2); neighbours.append(j)
						#gaussian kernel
						elif kl == 'gs':
							cnt += np.exp(-tmp**2/dc**2); neighbours.append(j)
						#normal distance
						elif kl == 'non': 
							cnt += 1; neighbours.append(j)
					if tmp > maxDist: maxDist = tmp
			wordParam[i] = [cnt, neighbours]
			if maxp < cnt: maxp = cnt; maxData = i
		#calculate delta as param[2], nearest higher density point j
		for i in range(0, len(dataVecs)):
			minDelta = maxDist
			affiliate = -1
			for j in range(0, len(dataVecs)):
				if wordParam[j][0] > wordParam[i][0]: 
					#euclidean distance
					tmp = np.linalg.norm(np.array(dataVecs[i]) - np.array(dataVecs[j]))
					if minDelta > tmp: 
						minDelta = tmp
						affiliate = j
			wordParam[i].extend([minDelta, affiliate])
		
		return wordParam

	def getBoarderDc(self, aWords, bWords, clusterWord, wordParams, a, b, dc, dataVecs, lasso):
		aBoarder = []
		bBoarder = []
		flag = False
		pa = wordParams[a][0]
		pb = wordParams[b][0]
		for word in aWords:
			wordNeighbour = wordParams[word][1]
			pword = wordParams[word][0]
			for cand in bWords:
				pcand = wordParams[cand][0]
				dist = stpw.euclidean(dataVecs[word], dataVecs[cand])
				if cand in wordNeighbour:
					if dist*pa/pword <= dc/lasso and dist*pb/pcand <= dc/lasso: 
						aBoarder.append(word) 
						bBoarder.append(cand)
		if len(aBoarder) != 0 : flag = True
		return aBoarder, bBoarder, flag

	def hasBoarder(self, aWords, bWords, wordParams):
		flag = False
		for a in aWords:
			neighbour = wordParams[a][1]
			for itm in neighbour:
				if itm in bWords:
					return True

	def calcHausdorffDist(self, aWords, bWords, a, b, dataVecs, wordParams):
		hausdorffDist = 0.
		if not self.hasBoarder(aWords, bWords, wordParams): return 2000
		supA2B = 0.
		for i in aWords:
			infA2B = 1000000.
			for j in bWords:
				tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
				if tmp < infA2B: infA2B = tmp
			if infA2B > supA2B: supA2B = infA2B
		supB2A = 0.
		for i in aWords:
			infB2A = 1000000.
			for j in bWords:
				tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
				if tmp < infB2A: infB2A = tmp
			if infB2A > supB2A: supB2A = infB2A
		hausdorffDist = max(supA2B, supB2A)
		
		return hausdorffDist

	def calcCompleteDist(self, aWords, bWords, a, b, dataVecs, wordParams):
		completeDist = 0.
		if not self.hasBoarder(aWords, bWords, wordParams): return 2000
		for i in aWords:
			for j in bWords:
				tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
				if tmp > completeDist: completeDist = tmp

		return completeDist

	def calcSingleDist(self, aWords, bWords, a, b, dataVecs, wordParams):
		dist = 1000
		if not self.hasBoarder(aWords, bWords, wordParams): return 2000
		for i in aWords:
			for j in bWords:
				tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
				if tmp < dist: dist = tmp
		return dist

	def calcAverageDist(self, aWords, bWords, a, b, dataVecs, wordParams):
		dist = 0.
		cnt = 0
		if not self.hasBoarder(aWords, bWords, wordParams): return 2000
		for i in aWords:
			for j in bWords:
				tmp = stpw.euclidean(dataVecs[i], dataVecs[j])
				dist += tmp
				cnt += 1
		dist = dist/cnt
		return dist

	#assign cluster in p's order, from high to low
	#if one cluster has a delta larger than dc, assign it a new cluster id
	def assignCluster(self, wordParams, centers, dataVecs, dc, label, iter, distType):
		boarders = set()
		clusterWord = {}
		maxArs = 0.
		maxAmi = 0.
		#coarsely assign cluster id based on centers
		pRank = [[wordParams[word][0], word] for word in wordParams]
		pRank.sort(reverse = True)
		wordCluster = {word:-1 for word in wordParams}
		id = 0
		centre2cluster = {}
		for p in pRank:
			if wordCluster[p[1]] == -1: 
				if p[1] in centers: wordCluster[p[1]] = p[1]; centre2cluster[p[1]] = p[1]
				else: 
					if wordParams[p[1]][3] == -1: 
						print('error, increase dc and try again....\n') 
						return maxArs, maxAmi, clusterWord
					wordCluster[p[1]] = wordCluster[wordParams[p[1]][3]]
		#merge false clusters

		cluster2centre = {centre2cluster[itm]: itm for itm in centre2cluster}
		for word in wordCluster:
			if wordCluster[word] in clusterWord: clusterWord[wordCluster[word]].append(word)
			else: clusterWord[wordCluster[word]] = [word]

		res = [wordCluster[itm] for itm in range(len(dataVecs))]
		maxArs = metrics.adjusted_rand_score(label, res)
		maxAmi = metrics.adjusted_mutual_info_score(label, res)

		tmpClusterWord = clusterWord
		cnt = 0
		
		while len(clusterWord) > 1 and cnt < iter:
			a = -1
			b = -1
			cnt += 1
			minDist = 1000
			for i in clusterWord:
				for j in clusterWord:
					if i == j: continue
					tmp = 0.
					if distType == 'single': tmp = self.calcSingleDist(clusterWord[i], clusterWord[j], i, j, dataVecs, wordParams)
					elif distType == 'complete': tmp = self.calcCompleteDist(clusterWord[i], clusterWord[j], i, j, dataVecs, wordParams)
					elif distType == 'average': tmp = self.calcAverageDist(clusterWord[i], clusterWord[j], i, j, dataVecs, wordParams)
					elif distType == 'hausdorff': tmp = self.calcHausdorffDist(clusterWord[i], clusterWord[j], i, j, dataVecs, wordParams)
					if tmp < minDist:
						minDist = tmp
						a = i
						b = j
			if a != -1 and b != -1:
				clusterWord[a].extend(clusterWord[b])
				clusterWord.pop(b)
			tmpWord2Cluster = {}
			for itm in clusterWord:
				for word in clusterWord[itm]:
					tmpWord2Cluster[word] = itm
			res = [tmpWord2Cluster[itm] for itm in range(len(dataVecs))]
			arsTmp = metrics.adjusted_rand_score(label, res)
			amiTmp = metrics.adjusted_mutual_info_score(label, res)
			if arsTmp > maxArs and amiTmp > maxAmi: 
				maxArs = arsTmp
				maxAmi = amiTmp
				tmpClusterWord = clusterWord
		return maxArs, maxAmi, tmpClusterWord

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

	def calcMaxDist(self, dataVecs, dc):
		rawDists = set()
		for i in range(len(dataVecs)):
			for j in range(i + 1, len(dataVecs)):
				dist = stpw.euclidean(dataVecs[i], dataVecs[j])
				rawDists.add(dist)
		dists = list(rawDists)
		dists.sort()
		return dists

	def getCenters(self, wordParams, dc):
		centers = []
		pRank = []
		for itm in wordParams:
			pRank.append([wordParams[itm][2], itm])
		pRank.sort()
		centers = [itm[1] for itm in pRank if itm[0] > dc and len(wordParams[itm[1]][1]) >= 3]
		return centers

	def plotClusterRes(self, wordCluster, dataVecs, centres, finalRes):
		tmpMap = {itm[1]: itm[0] for itm in enumerate(finalRes)}
		colors = cm.rainbow(np.linspace(0, 1, len(finalRes) + 1))
		dpcolorMap2 = [colors[tmpMap[wordCluster[itm]]] for itm in range(len(dataVecs))]
		'''
		for itm in centres:
			dpcolorMap2[itm] = 'k'
		'''
		dx = [dataVecs[i][0] for i in range(len(dataVecs))]
		dy = [dataVecs[i][1] for i in range(len(dataVecs))]
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=100)
		#plt.xlabel('X')
		#plt.ylabel('Y')
		plt.xticks([])
		plt.yticks([])
		plt.show()
		return

	def plotCluster(self, rawCluster, wordCluster, dx, dy, id, num, centers, boarders):
		colors = cm.rainbow(np.linspace(0, 1, 20))
		plt.figure(1)
		dpcolorMap1 = ['k' for i in range(len(dx))]
		for itm in centers:
			dpcolorMap1[itm] = 'r'
		for itm in boarders:
			if itm in centers: dpcolorMap1[itm] = 'g'
			else: dpcolorMap1[itm] = 'b'
		plt.scatter(dx, dy, c=dpcolorMap1)
		for i in range(len(dx)):
			plt.annotate(rawCluster[i], (dx[i],dy[i]))
		#dpcolorMap = [colors[wordCluster[i]] for i in range(len(dx))]
		plt.figure(2)
		dpcolorMap2 = ['k' for i in range(len(dx))]
		for itm in centers:
			dpcolorMap2[itm] = 'r'
		for itm in boarders:
			if itm in centers: dpcolorMap2[itm] = 'g'
			else: dpcolorMap2[itm] = 'b'
		plt.scatter(dx, dy, c=dpcolorMap2)
		for i in range(len(dx)):
			plt.annotate(wordCluster[i], (dx[i],dy[i]))
		plt.show()
		return
	
	#density peak clustering flow
	def run(self, dir, dc, lasso, kl):
		dx, dy, id, num = self.read(dir)
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		dists = self.calcMaxDist(dataVecs, dc)
		#realDc = dists[round(dc*len(dists))]
		realDc = dists[-1]*dc
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		#self.plotParams(wordParams)
		centers = self.getCenters(wordParams, realDc)
		rawCluster, wordCluster, boarders, finalRes, flag = self.assignCluster(wordParams, centers, dataVecs, realDc, lasso)
		self.plotClusterRes(wordCluster, dataVecs, centers, finalRes)
		label = [wordCluster[i] for i in wordCluster]
		return wordCluster, label

	#for evaluation
	def eval(self, dc, kl, dataVecs, label, iter, distType):
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, realDc)
		ars, ami, wordCluster = self.assignCluster(wordParams, centers, dataVecs, realDc, label, iter, distType)
		return ars, ami, centers, wordCluster


	def plotParams(self, params):
		p = []
		delta = []
		#x = p, y = delta
		for itm in params:
			p.append(params[itm][0])
			delta.append(params[itm][1])
		plt.plot(p, delta, 'ro')
		plt.show()
		return

if __name__ == "__main__":
	inst = huasdorffHier()
	
	'''
	data = []
	label = []
	num = 0
	print('name of the dataset: ')
	dataSet = input()
	#fp = open(dataSet+'.txt', 'w')
	dataTmp, label = loadData(dataSet)
	tmp = set(label)
	num = len(tmp)
	while True:
		print('Enter lasso:')
		lasso = float(input())
		inst.run('pathbased.txt', 0.08, lasso, 'nl')
	'''

	'''
	#done normal distance
	clusters = inst.run('Compound.txt', 0.06, 0.65, 'nl')
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
	distTmp = 'hausdorff'
	print(distTmp+" linkage")
	print('running hausdorff hierarchical merging......')
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
			 ['spiral.txt', 3, 'nl'],['R15.txt', 15, 'gs'],['D31.txt', 31, 'gs']]
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
		 ['spiral.txt', 3, 'nl']]
	dataScore = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}
	dataCnt = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}

	inst = huasdorffHier()
	for j in datasets:
		dx, dy, id, num = inst.read(j[0])
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		i = 0.01
		while i < 0.1:
			arsTmp, amidpc, centers, clusters = inst.eval(i, j[2], dataVecs, id, 20, distTmp)
			i += 0.01
			dataScore[j[0].split('.')[0]] += amidpc
			if amidpc != 0: dataCnt[j[0].split('.')[0]] += 1
	dataScore = {dataScore[itm]/9 for itm in dataScore}
	print(dataScore)
	print('The end...')
