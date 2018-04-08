import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn import metrics
from func import *

#read dataset
def read(dir):
	fp = open('C:/study/datasets/synthetic/' + dir)
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


def getNgbCell(cloc, clist):
	x = cloc[0]
	y = cloc[1]
	ncell = []
	n = len(clist)
	m = len(clist[0])
	ncell.extend([[x-1,y-2], [x,y-2], [x+1,y-2]])
	ncell.extend([[x-2,y-1], [x-1,y-1], [x,y-1], [x+1,y-1], [x+2,y-1]])
	ncell.extend([[x-2,y], [x-1,y], [x+1,y], [x+2,y]])
	ncell.extend([[x-2,y+1], [x-1,y+1], [x,y+1], [x+1,y+1], [x+2,y+1]])
	ncell.extend([[x-1,y+2], [x,y+2], [x+1,y+2]])
	newcell = []
	for itm in ncell:
		if itm[0] < 0 or itm[0] >= n: continue 
		if itm[1] < 0 or itm[1] >= m: continue
		newcell.append(itm)
	return newcell

class grid_mdpc:
	def __init__(self, feats, dc, k, maxdist):
		self.plist = []
		self.dlist = []
		self.ngblist = []
		self.k = k
		self.feats = feats
		self.labels = [-1 for itm in feats]
		self.dc = dc*maxdist
		self.centre = []
		self.grid = self.dc/np.sqrt(2)
		self.datatree = [[] for i in range(len(self.feats))]

	#build two-dimensional grid and calculate parameters
	def gridParams(self):
		'''
		build two dimensional grid, need to be extended to m-dimension later
		'''
		gridsize = self.grid
		featmax = [0 for i in range(len(self.feats[0]))]
		featmin = [10000000000 for i in range(len(self.feats[0]))]
		featloc = [[0,0] for i in range(len(self.feats))]
		for feat in self.feats:
			for idx in range(len(feat)):
				if feat[idx] > featmax[idx]: featmax[idx] = feat[idx]
				if feat[idx] < featmin[idx]: featmin[idx] = feat[idx]
		clist = [[[] for i in range(int(featmax[1]/gridsize) - int(featmin[1]/gridsize)+1)] for j in range(int(featmax[0]/gridsize)-int(featmin[0]/gridsize)+1)]
		for idx, feat in enumerate(self.feats):
			d0 = int(feat[0]/gridsize) - int(featmin[0]/gridsize)
			d1 = int(feat[1]/gridsize) - int(featmin[1]/gridsize)
			clist[d0][d1].append(idx)
			featloc[idx] = [d0, d1]
		
		'''
		calculate p
		'''
		for idx,feat in enumerate(self.feats):
			cloc = featloc[idx]
			dist = 0.
			ngb = []
			for itm in clist[cloc[0]][cloc[1]]:
				if itm == idx: continue
				dist += euclidean(self.feats[idx], self.feats[itm])
				ngb.append(itm)
			ncells = getNgbCell(cloc, clist)
			for cell in ncells:
				for itm in clist[cell[0]][cell[1]]:
					tmpdist = euclidean(self.feats[idx], self.feats[itm])
					if tmpdist <= self.dc: dist+= tmpdist; ngb.append(itm)
			self.plist.append(dist)
			self.ngblist.append(ngb)
		'''
		calculate delta
		'''
		clusterid = 0
		for idx, feat in enumerate(self.feats):
			cloc = featloc[idx]
			dist = 100000.
			deltaidx = -1
			for itm in clist[cloc[0]][cloc[1]]:
				if itm == idx: continue
				if self.plist[itm] >= self.plist[idx]:

					tt = euclidean(self.feats[idx],self.feats[itm])
					if dist > tt: dist = tt; deltaidx = itm
			ncells = getNgbCell(cloc, clist)
			for cell in ncells:
				for itm in clist[cell[0]][cell[1]]:
					if self.plist[itm] >= self.plist[idx]:
						tt = euclidean(feat[idx], feat[itm])
						if dist > tt: dist = tt; deltaidx = itm
			self.dlist.append([deltaidx, dist])
			datatree[deltaidx].append(idx)
			if deltaidx == -1: self.centre.append(idx); self.labels[idx] = clusterid; clusterid += 1
		return

	def clustering(self):
		#find seed clusters
		for centre in self.centre:
			idx = self.labels[centre]
			que = self.datatree[centre]
			i = 0
			j = len(que)
			while i != j:
				que.extend(self.datatree[que[i]])
				j += len(self.datatree[que[i]])
				self.labels[que[i]] = idx
				i += 1

		#find border data




		return

	#density peak clustering flow
	def run(self):
		self.gridParams()
		newlabels = self.clustering()
		return newlabels

if __name__ == "__main__":
	dx, dy, id, num = read('flame.txt')
	inst = grid_mdpc([[dx[i],dy[i]] for i in range(len(dx))],0.1,num, 29.2959041506)
	newlabels = inst.run()
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
	'''
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
		 ['spiral.txt', 3, 'nl'],['R15.txt', 15, 'gs'],['D31.txt', 31, 'gs']]
	datasets = [['pathbased.txt', 3, 'nl'],['Compound.txt', 6, 'nl'],['jain.txt', 2, 'nl'],['flame.txt', 2, 'nl'],['Aggregation.txt', 7, 'nl'],
		 ['spiral.txt', 3, 'nl']]
	i = 0.01
	dataScore = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}
	dataCnt = {'pathbased':0., 'Compound':0., 'jain':0., 'flame':0., 'Aggregation':0., 'spiral':0., 'R15':0., 'D31':0.}
	while dc < 0.11:
		for j in datasets:
			#print(j[0])
			dx, dy, id, num = read(j[0])
			inst = grid_mdpc(dc, [[dx[i],dy[i]] for i in range(len(dx))], num)
			inst.run()
			amidpc = 0.0
			amiTmp, centers, clusters = inst.run(j[0], i, 5, j[2])
			amidpc = amiTmp
			dataScore[j[0].split('.')[0]] += amidpc
			if amidpc != 0: dataCnt[j[0].split('.')[0]] += 1
		i += 0.01
	dataScore = {dataScore[itm]/9 for itm in dataScore}
	print(dataScore)
	'''
	print('The end...')