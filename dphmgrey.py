import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import stopWords as stpw
import re
from sklearn import metrics

class dpg:
	#calculate p and delta
	def calcParams(self, dataVecs, dc, kl):
		wordParam = {}
        #calculate p as param[0], neighbours as param[1]
		maxp = 0
		maxData = 0
		maxDist = 0.0
		for i in range(0, len(dataVecs)):
			cnt = dataVecs[i][2]
			neighbours = []
			for j in range(0, len(dataVecs)):
				if i!=j:
					tmp = stpw.euclidean(dataVecs[i][0:2], dataVecs[j][0:2])
					tmpDist = 0.
					if tmp <= dc: 
						#normal regularization
						if kl == 'nl':
							cnt += (1-(tmp**2/dc**2))*dataVecs[j][2]; neighbours.append(j)
						#gaussian kernel
						elif kl == 'gs':
							cnt += np.exp(-tmp**2/dc**2); neighbours.append(j)
						#normal distance
						elif kl == 'non': 
							cnt += 1; neighbours.append(j)
					if tmp > maxDist: maxDist = tmp
			wordParam[i] = [dataVecs[i][2], neighbours]
			if maxp < dataVecs[i][2]: maxp = dataVecs[i][2]; maxData = i
		#calculate delta as param[2], nearest higher density point j
		for i in range(0, len(dataVecs)):
			minDelta = maxDist
			affiliate = -1
			for j in range(0, len(dataVecs)):
				if wordParam[j][0] > wordParam[i][0]: 
					#euclidean distance
					tmp = np.linalg.norm(np.array(dataVecs[i][0:2]) - np.array(dataVecs[j][0:2]))
					if minDelta > tmp: 
						minDelta = tmp
						affiliate = j
			wordParam[i].extend([minDelta, affiliate])
		
		return wordParam

	def plotDG(self, wordParam, dx, dy):
		X = [wordParam[i][0] for i in wordParam]
		Y = [wordParam[i][2] for i in wordParam]
		pRank = []
		dpcolorMap1 = ['g' for i in range(len(wordParam))]
		for itm in wordParam:
			pRank.append([wordParam[itm][2] * wordParam[itm][0], itm])
		pRank.sort(reverse = True)
		centers = [itm[1] for itm in pRank[0:2]]
		S = [20 for i in range(len(dx))]
		plt.figure(1)
		dpcolorMap1 = [(0.3, 0.3, 0.3) for i in range(len(dx))]
		dpcolorMap2 = dpcolorMap1.copy()
		for cent in centers:
			dpcolorMap1[cent] = 'r'
			S[cent] = 100
		plt.scatter(X, Y, c=dpcolorMap1)
		plt.xlabel(r'$\rho$', fontsize=15)
		plt.ylabel(r'$\delta$', fontsize=15)
		centerX = []
		centerY = []
		for cent in centers:
			#dpcolorMap1[cent] = 'r'
			centerX.append(dx[cent])
			centerY.append(dy[cent])
		plt.figure(2)
		plt.scatter(dx, dy, c=dpcolorMap2, s=20, alpha=0.5)
		plt.xlabel('X', fontsize=15)
		plt.ylabel('Y', fontsize=15)

		#plt.scatter(centerX, centerY, c='r', s=50)
		plt.show()
		return

	def getBoarderDc(self, aWords, bWords, clusterWord, wordParams, a, b, dc, dataVecs, lasso):
		aBoarder = []
		bBoarder = []
		aB = []
		bB = []
		flag = False
		pa = wordParams[a][0]
		pb = wordParams[b][0]
		paAll = 0.
		pbAll = 0.
		for itm in aWords: paAll += wordParams[itm][0]
		for itm in bWords: pbAll += wordParams[itm][0]
		paAverage = paAll/len(aWords)
		pbAverage = pbAll/len(bWords)
		for word in aWords:
			wordNeighbour = wordParams[word][1]
			pword = wordParams[word][0]
			for cand in bWords:
				pcand = wordParams[cand][0]
				dist = stpw.euclidean(dataVecs[word][0:2], dataVecs[cand][0:2])
				if cand in wordNeighbour:
					aB.append(word) 
					bB.append(cand)
		if len(aB) == 0: return aB, bB, False
		paB = 0.
		pbB = 0.
		cntA = 0
		cntB = 0
		
		for itm in aB: 
			paB += wordParams[itm][0]
			cntA += 1
			for point in wordParams[itm][1]:
				if point in clusterWord[a]:
					paB += wordParams[point][0]
					cntA += 1
		for itm in bB: 
			pbB += wordParams[itm][0]
			cntB += 1
			for point in wordParams[itm][1]:
				if point in clusterWord[b]:
					pbB += wordParams[point][0]
					cntB += 1
		paBA = paB/cntA
		pbBA = pbB/cntB
		for word in aWords:
			wordNeighbour = wordParams[word][1]
			pword = wordParams[word][0]
			for cand in bWords:
				pcand = wordParams[cand][0]
				dist = stpw.euclidean(dataVecs[word][0:2], dataVecs[cand][0:2])
				if cand in wordNeighbour:
					if paAll > pbAll:
						if ((dist*((pa/pword)*lasso))/dc + (dist*((pb/pcand)*lasso))/dc)/2 * np.exp(
							min(
								abs(paAverage-pbAverage)/max(paAverage, pbAverage),
								abs(paBA-pbAll)/max(paBA, pbAll)))<= 1: 
							aBoarder.append(word) 
							bBoarder.append(cand)
							flag = True
					else:
						if ((dist*((pa/pword)*lasso))/dc + (dist*((pb/pcand)*lasso))/dc)/2 * np.exp(
							min(
								abs(paAverage-pbAverage)/max(paAverage, pbAverage),
								abs(pbBA-paAll)/max(pbBA, paAll)))<= 1: 
							aBoarder.append(word) 
							bBoarder.append(cand)
							flag = True
		if len(aBoarder) != 0 : flag = True
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
				aBoarder, bBoarder, hasBoarder = self.getBoarderDc(clusterWord[centre2cluster[i]], clusterWord[centre2cluster[j]], clusterWord, wordParams,
													 i, j, dc, dataVecs, lasso)
				#aBoarder, bBoarder, hasBoarder = self.getBoarder(clusterWord[centre2cluster[i]], clusterWord[centre2cluster[j]], clusterWord, wordParams)
				if hasBoarder:
					#distance of point 0-1 is (p0-p1)/dist(0,1)
					centreDistMat[i][j] = 1
					centreDistMat[j][i] = 1
					for itm in aBoarder: boarders.add(itm)
					for itm in bBoarder: boarders.add(itm)
		
		
		relation = {i: [] for i in centers}
		for centre in centers:
			maxDist = 0.
			parent = -1
			for centreDist in centreDistMat[centre]:
				if centreDistMat[centre][centreDist] > 0.: relation[centre].append(centreDist); relation[centreDist].append(centre)
		
		#print(relation)
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
				dist = stpw.euclidean(dataVecs[i][0:2], dataVecs[j][0:2])
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

	def plotCenter(self, rawCluster, wordCluster, dx, dy, id, num, centers, boarders, dc):
		colors = cm.rainbow(np.linspace(0, 1, 20))
		S = [100 for i in range(len(dx))]
		m = ['.' for i in range(len(dx))]
		plt.figure(1)
		dpcolorMap1 = ['k' for i in range(len(dx))]
		cx = []
		cy = []
		dpcolorMap2 = []
		for itm in centers:
			#dpcolorMap1[itm] = 'r'
			#S[itm] = 300
			m[itm] = '+'
			cx.append(dx[itm])
			cy.append(dy[itm])
			dpcolorMap2.append('r')
		plt.scatter(dx, dy, c=dpcolorMap1, s=S, marker='.', alpha=0.3)
		plt.scatter(cx, cy, c=dpcolorMap2, s=100, marker='*')
		plt.xticks([])
		plt.yticks([])
		#plt.xlabel('X',fontsize=15)
		#plt.ylabel('Y',fontsize=15)
		#plt.title('Histogram of IQ')
		plt.text(0, 6.15, r'$dc=$'+str(dc), fontsize=15, color='green')
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
		'''
		plt.figure(3)
		dpcolorMap3 = ['k' for i in range(len(dx))]
		for itm in centers:
			dpcolorMap3[itm] = 'r'
		for itm in boarders:
			if itm in centers: dpcolorMap3[itm] = 'g'
			else: dpcolorMap3[itm] = 'b'
		plt.scatter(dx, dy, c=dpcolorMap2)
		for idx, i in enumerate(range(len(dx))):
			plt.annotate(idx, (dx[i],dy[i]))
		'''
		plt.xticks([])
		plt.yticks([])
		plt.show()
		return



	#density peak clustering flow
	def run(self, dir, dc, lasso, kl):
		print(dir)
		dx, dy, id, num = self.read(dir)
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		dists = self.calcMaxDist(dataVecs, dc)
		#realDc = dists[round(dc*len(dists))]
		realDc = dists[-1]*dc
		print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		#self.plotDG(wordParams, dx, dy)
		centers = self.getCenters(wordParams, realDc)
		rawCluster, wordCluster, boarders, finalRes, valid = self.assignCluster(wordParams, centers, dataVecs, realDc, lasso)
		#self.plotCenter(rawCluster, wordCluster, dx, dy, id, num, centers, boarders, dc)
		#self.plotClusterRes(wordCluster, dataVecs, centers, finalRes)
		label = [wordCluster[i] for i in range(len(dataVecs))]
		arsTmp = metrics.adjusted_rand_score(id, label)
		amiTmp = metrics.adjusted_mutual_info_score(id, label)
		print(arsTmp)
		print(amiTmp)
		return wordCluster, label

	#for evaluation
	def eval(self, dc, lasso, kl, dataVecs):
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, realDc)
		rawCluster, wordCluster, boarders, finalRes, valid = self.assignCluster(wordParams, centers, dataVecs, realDc, lasso)
		#self.plotClusterRes(wordCluster, dataVecs, centers, finalRes)
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
	inst = dp()
	#clusters = inst.run('Compound.txt', 0.05, 5, 'nl')
	#inst.run('flame.txt', 0.1, 0.65, 'nl')
	#inst.run('horn.txt', 0.05, 0.15, 'nl')
	#inst.run('tmp.txt', 0.015, 0, 'nl')
	#clusters = inst.run('Aggregation.txt', 0.1, 1.5, 'nl')
	#clusters = inst.run('D31.txt', 0.01, 0.2, 'gs')
	#clusters = inst.run('jain.txt', 0.1, 0.5, 'nl')

	clusters = inst.run('pathbased.txt', 0.08, 0.6, 'nl')
	'''
	#done normal distance
	clusters = inst.run('jain.txt', 0.10, 0.1, 'nl')
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
	'''
	#done normal distance
	#clusters = inst.run('jain.txt', 0.10, 0.1, 'nl')
	#clusters = inst.run('Compound.txt', 0.1, 1, 'nl')
	#clusters = inst.run('pathbased.txt', 0.085, 0.2, 'nl')
	#clusters = inst.run('Aggregation.txt', 0.09, 2, 'nl')
	#clusters = inst.run('spiral.txt', 0.12, 1, 'nl')
	#inst.run('flame.txt', 0.2, 1.3, 'nl')
	#inst.run('a3.txt', 0.020, 1.5, 'nl')

	#done Gaussian distance
	#clusters = inst.run('R15.txt', 0.040, 1, 'gs')
	#inst.run('s1.txt', 0.035, 0.7, 'gs')
	#clusters = inst.run('D31.txt', 0.021, 1, 'gs')
	#clusters = inst.run('s4.txt', 0.06, 7, 'gs')
	'''
	inst.run('flame.txt', 0.1, 0.65, 'nl')
	clusters = inst.run('pathbased.txt', 0.085, 0.6, 'nl')
	clusters = inst.run('jain.txt', 0.10, 0.5, 'nl')
	clusters = inst.run('Aggregation.txt', 0.05, 1.5, 'nl')
	clusters = inst.run('Compound.txt', 0.05, 1, 'nl')
	clusters = inst.run('spiral.txt', 0.06, 0.5, 'nl')
	#inst.run('a3.txt', 0.020, 1.5, 'nl')

	#done Gaussian distance
	clusters = inst.run('R15.txt', 0.040, 1, 'gs')
	#inst.run('s1.txt', 0.035, 0.7, 'gs')
	clusters = inst.run('D31.txt', 0.021, 1, 'gs')
	#clusters = inst.run('s4.txt', 0.06, 7, 'gs')

	print('The end...')