import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics

class dpha:
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
					if tmp <= dc: 
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

	def calcDPHADist(self, aWords, bWords, clusterWord, wordParams, a, b, dc, dataVecs):
		pa = wordParams[a][0]
		pb = wordParams[b][0]
		flag = False
		paAll = 0.
		pbAll = 0.
		for itm in aWords: paAll += wordParams[itm][0]
		for itm in bWords: pbAll += wordParams[itm][0]
		paAverage = paAll/len(aWords)
		pbAverage = pbAll/len(bWords)
		for word in aWords:
			wordNeighbour = wordParams[word][1]
			for neighbour in wordNeighbour:
				if neighbour in bWords: flag = True; break
		if flag == False: return 1000.
		paSingle = []
		for itm in aWords: 
			paB =0.
			cntA = 0
			paB += wordParams[itm][0]
			cntA += 1
			for point in wordParams[itm][1]:
				if point in clusterWord[a]:
					paB += wordParams[point][0]
					cntA += 1
			paSingle.append(paB/cntA)
		pbSingle = []
		for itm in bWords: 
			pbB =0.
			cntB =0
			pbB += wordParams[itm][0]
			cntB += 1
			for point in wordParams[itm][1]:
				if point in clusterWord[b]:
					pbB += wordParams[point][0]
					cntB += 1
			pbSingle.append(pbB/cntB)
		denseDist=1000.
		for wid, word in enumerate(aWords):
			pword = wordParams[word][0]
			wordNeighbour = wordParams[word][1]
			for cid, cand in enumerate(bWords):
				if cand not in wordNeighbour: continue
				pcand = wordParams[cand][0]
				dist = stpw.euclidean(dataVecs[word], dataVecs[cand])
				if paAll > pbAll:
					tmp = ((dist*pa/(pword*dc))/2 + (dist*pb/(pcand*dc))/2) / np.exp(
						min(
							abs(paAverage-pbAverage)/max(paAverage, pbAverage),
							abs(paSingle[wid]-pbAll)/max(paSingle[wid], pbAll)))
				else:
					tmp = ((dist*pa/(pword*dc))/2 + (dist*pb/(pcand*dc))/2) / np.exp(
						min(
							abs(paAverage-pbAverage)/max(paAverage, pbAverage),
							abs(pbSingle[cid]-paAll)/max(pbSingle[cid], paAll)))
				if denseDist > tmp: denseDist = tmp
		return denseDist

	#assign cluster in p's order, from high to low
	#if one cluster has a delta larger than dc, assign it a new cluster id
	def assignCluster(self, wordParams, centers, dataVecs, dc, label, iter):
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
		distList = []
		#distRecord = []
		for i in clusterWord:
			for j in clusterWord:
				if i == j: continue
				tmp = self.calcDPHADist(clusterWord[i], clusterWord[j], clusterWord, wordParams, i, j, dc, dataVecs)
				#distRecord.append([tmp,i,j])
				if tmp not in distList: distList.append(tmp)
		distList.sort()
		#print(distList)
		oldClusterWord = clusterWord.copy()

		relation = {i: [] for i in centers}
		for dist in distList:
			a = -1
			b = -1
			cnt += 1
			#print("dist = "+str(dist))
			for i in oldClusterWord:
				for j in oldClusterWord:
					if i == j: continue
					tmp = self.calcDPHADist(oldClusterWord[i], oldClusterWord[j], clusterWord, wordParams, i, j, dc, dataVecs)
					if dist == 1000.: continue
					if tmp <= dist:
						a = i
						b = j
						if a != -1 and b != -1:
							relation[a].append(b)
							relation[b].append(a)
				#print('merged cluster is: '+str(b)+' with: '+str(a))
			mergedList = []
			#print(relation)
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
			#print(mergedList)
			clusterRel = {i: -1 for i in clusterWord.keys()}
			for merge in mergedList:
				for itm in merge:
					clusterRel[centre2cluster[itm]] = centre2cluster[merge[0]]

			realCluster = {word:-1 for word in wordParams}
			#self.plotClusterRes(wordCluster, dataVecs, centers, wordCluster, "test", 5)
			for word in wordCluster:
				if clusterRel[wordCluster[word]] != -1: realCluster[word] = cluster2centre[clusterRel[wordCluster[word]]]
				else: realCluster[word] = cluster2centre[wordCluster[word]]
			res = []
			for itm in range(len(realCluster)): res.append(realCluster[itm])
			#self.plotClusterRes(realCluster, dataVecs, centers, realCluster, "test", 5)
			arsTmp = metrics.adjusted_rand_score(label, res)
			amiTmp = metrics.adjusted_mutual_info_score(label, res)
			if arsTmp > maxArs and amiTmp > maxAmi: 
				maxArs = arsTmp
				maxAmi = amiTmp
				tmpClusterWord = clusterWord
		return maxArs, maxAmi, tmpClusterWord

	def plotClusterRes(self, wordCluster, dataVecs, centres, finalRes, dir, num):
		colors = cm.rainbow(np.linspace(0, 1, len(finalRes) + 1))
		tmp = {}
		for i in range(num+1):
			if i%2 ==1: tmp[i] = i-1
			else: tmp[i] = i+4
		dpcolorMap2 = [colors[wordCluster[itm]] for itm in range(len(dataVecs))]

		dx = [dataVecs[i][0] for i in range(len(dataVecs))]
		dy = [dataVecs[i][1] for i in range(len(dataVecs))]		
		
		cx = []
		cy = []
		centreMap = ['r' for itm in centres]
		for itm in centres:
			cx.append(dx[itm])
			cy.append(dy[itm])
		plt.scatter(cx, cy, c=centreMap, marker='+', s=250, zorder=2)

		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=100, edgecolor='g', alpha=0.8, zorder=1)
		#plt.xlabel('X')
		#plt.ylabel('Y')
		plt.xticks([])
		plt.yticks([])
		plt.savefig("C:\\study\\8data\\"+dir.split('.')[0]+"_dpha.png", bbox_inches='tight', pad_inches = 0)
		plt.show()
		return


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


	#density peak clustering flow
	def run(self, dir, dc, lasso, kl):
		dx, dy, id, num = self.read(dir)
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		dists = self.calcMaxDist(dataVecs, dc)
		#realDc = dists[round(dc*len(dists))]
		realDc = dists[-1]*dc
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		#self.plotDG(wordParams, dx, dy)
		centers = self.getCenters(wordParams, realDc)
		ars, ami, wordCluster = self.assignCluster(wordParams, centers, dataVecs, realDc, id, iter)
		#self.plotCluster(rawCluster, wordCluster, dx, dy, id, num, centers, boarders)
		#self.plotCenter(rawCluster, wordCluster, dx, dy, id, num, centers, boarders, dc, kl)
		#self.plotClusterRes(wordCluster, dataVecs, centers, finalRes, dir, num)
		#print(ami)
		return ars, ami, centers, wordCluster

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
	inst = dpha()
	#correct
	#inst.run('flame.txt', 0.1, 0.65, 'nl')
	#clusters = inst.run('pathbased.txt', 0.085, 0.6, 'nl')
	#clusters = inst.run('jain.txt', 0.10, 0.5, 'nl')
	#clusters = inst.run('Aggregation.txt', 0.05, 1.5, 'nl')
	#clusters = inst.run('Compound.txt', 0.05, 1, 'nl')
	#clusters = inst.run('spiral.txt', 0.06, 0.5, 'nl')
	#inst.run('a3.txt', 0.020, 1.5, 'nl')

	#done Gaussian distance
	#clusters = inst.run('R15.txt', 0.040, 1000, 'gs')
	#inst.run('s1.txt', 0.035, 0.7, 'gs')
	#clusters = inst.run('D31.txt', 0.021, 10000, 'gs')
	#clusters = inst.run('s4.txt', 0.06, 7, 'gs')
	#end
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
		 ['spiral.txt', 3, 'nl'],['R15.txt', 15, 'gs'],['D31.txt', 31, 'gs']]
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
		 ['spiral.txt', 3, 'nl']]
	i = 0.01
	dataScore = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}
	dataCnt = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}
	while i < 0.11:
		for j in datasets:
			#print(j[0])
			dx, dy, id, num = inst.read(j[0])
			dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
			arsdpc = 0.0
			amidpc = 0.0
			arsTmp, amiTmp, centers, clusters = inst.run(j[0], i, 5, j[2])
			amidpc = amiTmp
			dataScore[j[0].split('.')[0]] += amidpc
			if amidpc != 0: dataCnt[j[0].split('.')[0]] += 1
		i += 0.01
	dataScore = {dataScore[itm]/9 for itm in dataScore}
	print(dataScore)
	print('The end...')