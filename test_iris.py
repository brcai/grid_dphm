from density_peak_clustering import dp
from dpmc_hausdorff import dph
from ddbscan import ddbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from read_data import readDataFile, loadData, scalarTmp
from sklearn import metrics
from dp_orig_bk import dpOrig
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.lines as mlines


if __name__ == '__main__':
	# import some data to play with
	data, y = loadData('iris')	
	X = PCA(n_components=3).fit_transform(data)
	num = 3
	'''
	print('running density peak clustering......')
	dc = 0.08
	inst = dpOrig()
	dlabel, valid, centers = inst.eval(dc, num, 'nl', X)
	print(metrics.adjusted_rand_score(y, dlabel))
	print(metrics.adjusted_mutual_info_score(y, dlabel))
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	a1 = [itm[0] for itm in X]
	a2 = [itm[1] for itm in X]
	a3 = [itm[2] for itm in X]
	
	ax.scatter(a1, a2, a3, c=dlabel,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
	'''
	print('running density peak merge clustering......')
	dmc = 0.08
	lasso = 0.6
	inst = dpOrig()
	#inst = dp()
	#dmlabel, valid, centers = inst.eval(dmc, lasso, 'nl', X)
	dmlabel, valid, centers = inst.eval(dmc, 3, 'nl', X)
	#dbs = DBSCAN(eps=0.2, min_samples=1).fit(X)
	#dmlabel = dbs.labels_
	print(metrics.adjusted_rand_score(y, dmlabel))
	print(metrics.adjusted_mutual_info_score(y, dmlabel))

	fig = plt.figure(2, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)

	a1 = [itm[0] for itm in X]
	a2 = [itm[1] for itm in X]
	a3 = [itm[2] for itm in X]
	lset = set(dmlabel)
	board = ['r', 'g', 'y']
	lmap = {idx: i for idx, i in enumerate(lset)}
	'''
	red_patch = mlines.Line2D([], [], color='white',marker='s', markeredgecolor='k',
                          markersize=10, label='setosa')
	green_patch = mlines.Line2D([], [], color='white',marker='v', markeredgecolor='b',
                          markersize=10, label='versicolor')
	yellow_patch = mlines.Line2D([], [], color='white',marker='^', markeredgecolor='m',
                          markersize=10, label='virginica')
	'''
	markerList = []
	for idx, itm in enumerate(dmlabel):
		color = 'white'
		al = '1'
		if itm != y[idx]: color = 'r'
		if itm == 1: ax.scatter(a1[idx], a2[idx], a3[idx], c=color,edgecolor='k', marker='s', s=80)
		elif itm == 2: ax.scatter(a1[idx], a2[idx], a3[idx], c=color,edgecolor='b', marker='v', s=80)
		else: ax.scatter(a1[idx], a2[idx], a3[idx], c=color,edgecolor='m', marker='^',  s=80)
	#plt.legend(handles=[red_patch, green_patch, yellow_patch])
	colorMap = [board[lmap[itm]] for itm in dmlabel]
	plt.savefig("C:\\study\\iris\\org.png", bbox_inches='tight', pad_inches = 0)
	'''
	fig = plt.figure(3, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	a1 = [itm[0] for itm in X]
	a2 = [itm[1] for itm in X]
	a3 = [itm[2] for itm in X]
	lset = set(y)
	lmap = {idx: i for idx, i in enumerate(lset)}
	board = ['r', 'g', 'y']
	red_patch = mpatches.Patch(color='r', label='setosa')
	green_patch = mpatches.Patch(color='g', label='versicolor')
	yellow_patch = mpatches.Patch(color='y', label='virginica')
	plt.legend(handles=[red_patch, green_patch, yellow_patch])

	colorMap = [board[lmap[itm]] for itm in y]
	ax.scatter(a1, a2, a3, c=colorMap, edgecolor='k',s=50)
	'''

	'''
	print('running dbscan......')
	fig = plt.figure(4, figsize=(8, 6))
	dbs = DBSCAN(eps=0.1, min_samples=1).fit(X)
	dblabel = dbs.labels_
	ax = Axes3D(fig, elev=-150, azim=110)
	a1 = [itm[0] for itm in X]
	a2 = [itm[1] for itm in X]
	a3 = [itm[2] for itm in X]

	ax.scatter(a1, a2, a3, c=dblabel,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
	print(metrics.adjusted_rand_score(y, dblabel))
	print(metrics.adjusted_mutual_info_score(y, dblabel))

	print('running hausdorff clustering......')
	dmc = 0.08
	lasso = 0.13
	inst = dph()
	dphlabel, valid, centers = inst.eval(dmc, lasso, 'nl', X)
	print(metrics.adjusted_rand_score(y, dphlabel))
	print(metrics.adjusted_mutual_info_score(y, dphlabel))

	fig = plt.figure(5, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	a1 = [itm[0] for itm in X]
	a2 = [itm[1] for itm in X]
	a3 = [itm[2] for itm in X]
	ax.scatter(a1, a2, a3, c=dphlabel,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

	'''
	plt.show()

	print("End of Test!")

