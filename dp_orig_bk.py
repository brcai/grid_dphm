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

class dpOrig:
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

	def assignCluster(self, wordParams, centers, dc, lasso):
		pRank = [[wordParams[word][0], word] for word in wordParams]
		pRank.sort(reverse = True)
		wordCluster = {word:-1 for word in wordParams}
		id = 0
		centre2cluster = {}
		for p in pRank:
			if wordCluster[p[1]] == -1: 
				if p[1] in centers: wordCluster[p[1]] = id; centre2cluster[p[1]] = id; id += 1
				else: 
					if wordParams[p[1]][3] == -1: 
						#print('error, increase dc and try again....\n') 
						return wordCluster, False
					wordCluster[p[1]] = wordCluster[wordParams[p[1]][3]]
		return wordCluster, True

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

	def getCenters(self, wordParams, num, dc):
		centers = []
		pRank = []
		for itm in wordParams:
			pRank.append([wordParams[itm][2] * wordParams[itm][0], itm])
		pRank.sort(reverse = True)
		if num > len(pRank): return centers
		centers = [itm[1] for itm in pRank[0:num]]
		return centers

	def plotClusterRes(self, wordCluster, dx, dy, lasso, centers, dir):
		colors = cm.rainbow(np.linspace(0, 1, lasso + 1))
		dpcolorMap2 = [colors[wordCluster[itm]] for itm in range(len(dx))]
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=150, alpha=0.8, zorder=1, edgecolor='k')

		cx = []
		cy = []
		for itm in centers:
			cx.append(dx[itm])
			cy.append(dy[itm])
		#plt.scatter(cx, cy, c='r',marker='+', s=400, zorder=2)
		plt.text(14.5, 1.0, r'$NMI=$'+str(0.5038), fontsize=15, color='black')
		#plt.xlabel('X', fontsize=25)
		#plt.ylabel('Y', fontsize=25)
		plt.xticks([])
		plt.yticks([])
		plt.axis('off')
		#plt.savefig("C:\\study\\8data\\Compound_org.png", bbox_inches='tight', pad_inches = 0)
		plt.savefig("C:\\study\\8data\\"+dir.split('.')[0]+"_dp.png", bbox_inches='tight', pad_inches = 0)
		plt.show()
		return

	def run(self, dir, dc, lasso, kl):
		dx, dy, id, num = self.read(dir)
		#colors = cm.rainbow(np.linspace(0, 1, num + 1))
	
		dpcolorMap2 = ['k' for itm in range(len(dx))]
		plt.scatter(dx, dy, c=dpcolorMap2, marker='.', s=100)
		#plt.xlabel('X', fontsize=25)
		#plt.ylabel('Y', fontsize=25)
		plt.xticks([])
		plt.yticks([])
		plt.axis('off')
		#plt.savefig("C:\\study\\dgf\\dgf_org_2.png",bbox_inches='tight', pad_inches = 0)
		#plt.show()
		arsTmp = 0.
		amiTmp = 0.
		
		dataVecs = [[dx[i], dy[i]] for i in range(len(dx))]
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		#print('dc = ' + str(dc)+' realDc = '+str(realDc))
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, lasso, realDc)
		#self.plotDgraph(wordParams, centers)
		if len(centers) == 0: return wordCluster, label, arsTmp, amiTmp
		wordCluster, valid = self.assignCluster(wordParams, centers, realDc, lasso)
		self.plotClusterRes(wordCluster, dx, dy, lasso, centers, dir)
		label = [wordCluster[i] for i in range(len(dataVecs))]
		arsTmp = metrics.adjusted_rand_score(id, label)
		amiTmp = metrics.adjusted_mutual_info_score(id, label)
		print(amiTmp)
		return wordCluster, label, arsTmp, amiTmp

	def plotDgraph(self, wordParams, centers):
		dx = [wordParams[itm][0] for itm in wordParams]
		dy = [wordParams[itm][2] for itm in wordParams]
		dpcolorMap3 = ['k' for i in range(len(dx))]
		cx = []
		cy = []
		for center in centers:
			cx.append(dx[center])
			cy.append(dy[center])
		ident = 0	
		
		for center in centers:
			dx.pop(center)
			dy.pop(center)
			ident += 1
		
		plt.scatter(dx, dy, c=dpcolorMap3, marker='.', s=300, alpha=0.5)
		dpcolorMap1 = ['r']
		plt.scatter(cx, cy, c=dpcolorMap1, marker='.', s=1000,edgecolor='k')
		plt.xlabel(r'$\rho$', fontsize=25)
		plt.ylabel(r'$\delta$', fontsize=25)
		
		plt.savefig("C:\\study\\dgf\\dgf_dg_2.png",bbox_inches='tight', pad_inches = 0)
		plt.show()
		return

	#for evaluation
	def eval(self, dc, lasso, kl, dataVecs):
		dists = self.calcMaxDist(dataVecs, dc)
		realDc = dists[-1]*dc
		wordParams = self.calcParams(dataVecs, realDc, kl)
		centers = self.getCenters(wordParams, lasso, realDc)
		#self.plotDgraph(wordParams, centers)
		wordCluster, valid = self.assignCluster(wordParams, centers, realDc, lasso)
		#self.plotClusterRes(wordCluster, dx, dy, lasso)
		label = [wordCluster[i] for i in wordCluster]
		return label, valid, centers


if __name__ == "__main__":
	inst = dpOrig()
	#inst.run('tmp.txt', 0.03, 10, 'nl')
	#inst.run('dgf_2.txt', 0.1, 2, 'nl')
	#inst.run('D31.txt', 0.02, 31, 'nl')
	#clusters = inst.run('jain.txt', 0.06, 2, 'nl')
	#a, b, tmpArs, tmpAmi = inst.run('Compound.txt', 0.01, 5, 'nl')
	#print(tmpArs)
	#clusters = inst.run('pathbased.txt', 0.06, 3, 'nl')
	#clusters = inst.run('Aggregation.txt', 0.1, 7, 'nl')
	#a,b,c,d = inst.run('tmp.txt', 0.05, 5, 'nl')
	#print(c)
	#print(d)
	'''
	datasets = [['R15.txt', 15, 'gs'],['D31.txt', 31, 'gs'], ['Compound.txt', 5, 'nl'],['pathbased.txt', 3, 'nl'],['jain.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],['flame.txt', 2, 'nl'],
			 ['spiral.txt', 3, 'nl']]
	#done normal distance
	for j in datasets:
		print(j[0])
		maxDc = 0.
		maxAmi = 0.
		maxArs = 0.
		for i in frange(0.05, 0.2, 0.05):
			a, b, tmpArs, tmpAmi = inst.run(j[0], i, j[1], j[2])
			if maxAmi < tmpAmi and maxArs < tmpArs:
				maxAmi = tmpAmi
				maxArs = tmpArs
				maxDc = i
		print(maxArs)
		print(maxAmi)
		print(maxDc)
	'''
	#inst.run('flame.txt', 0.1, 2, 'nl')
	#clusters = inst.run('pathbased.txt', 0.06, 3, 'nl')
	clusters = inst.run('jain.txt', 0.06, 2, 'nl')
	#clusters = inst.run('Compound.txt', 0.1, 5, 'nl')
	
	#clusters = inst.run('Aggregation.txt', 0.1, 7, 'nl')
	#clusters = inst.run('spiral.txt', 0.1, 3, 'nl')
	#inst.run('flame.txt', 0.15, 2, 'nl')
	#inst.run('a3.txt', 0.020, 50, 'nl')

	#done Gaussian distance
	#clusters = inst.run('R15.txt', 0.04, 15, 'gs')
	#inst.run('s1.txt', 0.028, 15, 'gs')
	#clusters = inst.run('D31.txt', 0.021, 31, 'gs')
	#clusters = inst.run('s4.txt', 0.06, 15, 'gs')
	print('The end...')
