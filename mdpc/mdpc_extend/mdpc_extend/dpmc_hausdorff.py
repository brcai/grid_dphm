import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import stopWords as stpw
import re
from sklearn import metrics
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


class dph:
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
		'''				
		weight = 0.
		pSmall = pa if pa < pb else pb
		for itm in aBoarder: weight += wordParams[itm][0]
		for itm in bBoarder: weight += wordParams[itm][0]
		if len(aBoarder) != 0 and weight >= pSmall : flag = True
		'''
		if len(aBoarder) != 0 : flag = True
		return aBoarder, bBoarder, flag

	def calcHausdorffDist(self, aWords, bWords, a, b, dataVecs, lasso):
		aBoarder = []
		bBoarder = []
		flag = False
		hausdorffDist = 0.
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
		if hausdorffDist < lasso: flag = True
		return aBoarder, bBoarder, flag

	#assign cluster in p's order, from high to low
	#if one cluster has a delta larger than dc, assign it a new cluster id
	def assignCluster(self, wordParams, centers, dataVecs, dc, lasso):
		boarders = set()
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
						#print('error, increase dc and try again....\n') 
						return wordCluster, [], [], [], False
					wordCluster[p[1]] = wordCluster[wordParams[p[1]][3]]
		#merge false clusters
		clusterWord = {}
		cluster2centre = {centre2cluster[itm]: itm for itm in centre2cluster}
		for word in wordCluster:
			if wordCluster[word] in clusterWord: clusterWord[wordCluster[word]].append(word)
			else: clusterWord[wordCluster[word]] = [word]
		mergedCluster = {id: [] for id in clusterWord}
		centreDistMat = {i: {j: 0. for j in centers if j != i} for i in centers}
		for i in centers:
			for j in centers:
				if  i == j or centreDistMat[i][j] != 0: continue
				aBoarder, bBoarder, hasBoarder = self.calcHausdorffDist(clusterWord[centre2cluster[i]], clusterWord[centre2cluster[j]], i, j, dataVecs, lasso)
				#aBoarder, bBoarder, hasBoarder = self.getBoarderDc(clusterWord[centre2cluster[i]], clusterWord[centre2cluster[j]], clusterWord, wordParams,
				#									 i, j, dc, dataVecs, lasso)
				#aBoarder, bBoarder, hasBoarder = self.getBoarder(clusterWord[centre2cluster[i]], clusterWord[centre2cluster[j]], clusterWord, wordParams)
				if hasBoarder:
					#distance of point 0-1 is (p0-p1)/dist(0,1)
					centreDistMat[i][j] = 1
					centreDistMat[j][i] = 1
		
		relation = {i: [] for i in centers}
		for centre in centers:
			maxDist = 0.
			parent = -1
			for centreDist in centreDistMat[centre]:
				if centreDistMat[centre][centreDist] > 0.: relation[centre].append(centreDist); relation[centreDist].append(centre)
		
		mergedList = []
		visited = {id: -1 for id in centers}
		for id in centers:
			if visited[id] == -1:
				visited[id] = 1
				mergedSet = set()
				if len(relation[id]) != 0:
					mergedSet = set(relation[id])
					mergedSet.add(id)
				else: continue
				que = mergedSet.copy()
				while len(que) != 0:
					newQue = set()
					for link in que:
						if visited[link] == 1: continue
						visited[link] = 1
						mergedSet.add(link)
						if len(relation[link]) != 0:
						   for itm in relation[link]: [newQue.add(itm) for itm in relation[link] if itm not in mergedSet]
					que = newQue.copy()
				tmpList = []
				[tmpList.append(itm) for itm in mergedSet]
				tmpList.sort()
				mergedList.append(tmpList)
		clusterRel = {i: -1 for i in clusterWord.keys()}
		for merge in mergedList:
			for itm in merge:
				clusterRel[centre2cluster[itm]] = centre2cluster[merge[0]]

		realCluster = {word:-1 for word in wordParams}
		for word in wordCluster:
			if clusterRel[wordCluster[word]] != -1: realCluster[word] = cluster2centre[clusterRel[wordCluster[word]]]
			else: realCluster[word] = cluster2centre[wordCluster[word]]
		realWord = {}
		for word in realCluster:
			if realCluster[word] in realWord: realWord[realCluster[word]].append(word)
			else: realWord[realCluster[word]] = [word]
		finalRes = {}
		finalCluster = {}
		idx = 0
		for itm in realWord:
			finalRes[idx] = realWord[itm]
			idx += 1
		#print('number of clusters is : '+ str(idx))
		for itm in finalRes:
			for elem in finalRes[itm]:
				finalCluster[elem] = itm
		#return wordCluster, realCluster, boarders
		return wordCluster, finalCluster, boarders, finalRes, True

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
		colors = cm.rainbow(np.linspace(0, 1, len(finalRes) + 1))
		dpcolorMap2 = [colors[wordCluster[itm]] for itm in range(len(dataVecs))]
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
		print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		#self.plotParams(wordParams)
		centers = self.getCenters(wordParams, realDc)
		rawCluster, wordCluster, boarders, finalRes, flag = self.assignCluster(wordParams, centers, dataVecs, realDc, lasso)
		label = [wordCluster[i] for i in wordCluster]
		arsTmp = metrics.adjusted_rand_score(id, label)
		amiTmp = metrics.adjusted_mutual_info_score(id, label)
		print(arsTmp)
		print(amiTmp)
		self.plotClusterRes(wordCluster, dataVecs, centers, finalRes)

		return wordCluster, label, arsTmp, amiTmp

	#for evaluation
	def eval(self, dc, thred, kl, dataVecs):
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		lasso = dists[-1]*thred
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, realDc)
		rawCluster, wordCluster, boarders, finalRes, valid = self.assignCluster(wordParams, centers, dataVecs, realDc, lasso)
		label = [wordCluster[i] for i in range(len(wordCluster))]
		return label, valid, centers


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
	inst = dph()
	'''
	datasets = [['flame.txt', 2, 'nl'],['jain.txt', 2, 'nl'],['Compound.txt', 6, 'nl'],['pathbased.txt', 3, 'nl'],['Aggregation.txt', 7, 'nl'],
			 ['spiral.txt', 3, 'nl'],['R15.txt', 15, 'gs'],['D31.txt', 31, 'gs']]
	#done normal distance
	for j in datasets:
		print(j[0])
		maxDc = 0.
		maxAmi = 0.
		maxArs = 0.
		for i in frange(0.1, 1, 0.05):
			a, b, tmpArs, tmpAmi = inst.run(j[0], i, j[1], j[2])
			if maxAmi < tmpAmi and maxArs < tmpArs:
				maxAmi = tmpAmi
				maxArs = tmpArs
				maxDc = i
		print(maxArs)
		print(maxAmi)
		print(maxDc)
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
	while True:
		print('Enter lasso:')
		lasso = float(input())
		inst.run('jain.txt', 0.10, lasso, 'nl')
	'''
	clusters,label,arsTmp, amiTmp = inst.run('jain.txt', 0.010, 0.6, 'nl')
	clusters = inst.run('Compound.txt', 0.06, 0.65, 'nl')
	clusters = inst.run('pathbased.txt', 0.085, 0.8, 'nl')
	clusters = inst.run('Aggregation.txt', 0.036, 1, 'nl')
	clusters = inst.run('spiral.txt', 0.12, 1, 'nl')
	inst.run('flame.txt', 0.1, 1, 'nl')
	clusters = inst.run('R15.txt', 0.040, 1, 'gs')
	clusters = inst.run('D31.txt', 0.021, 1, 'gs')
	

	print('The end...')
