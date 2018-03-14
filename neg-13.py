import re
import numpy
from density_peak_clustering import dp
from dp_orig_bk import dpOrig
from read_data import scalarTmp
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.colors as matcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == "__main__":
	weak = []
	image = read_pgm("C:\\Users\\bcai\\Downloads\\CarData.tar\\CarData\\TrainImages\\neg-13.pgm", byteorder='<')
	for ldx, line in enumerate(image):
		tmp = []
		for idx, itm in enumerate(line):
			if itm < 170 :
				tmp.append(0)
			else:
				tmp.append(255)
		weak.append(tmp)
	plt.figure(1)
	#plt.imshow(image, cmap='jet')
	#plt.savefig("C:\\study\\image\\13.png",bbox_inches='tight', pad_inches = 0)
	plt.figure(2)
	pre = [[np.abs(i[j]-255) for j in range(len(i))] for i in weak]
	plt.imshow(pre, cm.gray)
	plt.savefig("C:\\study\\image\\13_pre.png",bbox_inches='tight', pad_inches = 0)
	#plt.show()
	plt.figure(3)
	dataTmp = []
	for i in range(len(weak)):
		tmp = []
		for j in range(len(weak[0])):
			if weak[i][j] != 0: tmp = [i,j]; dataTmp.append(tmp)
	
	tmpx = [data[0] for data in dataTmp]
	tmpy = [data[1] for data in dataTmp]


	#inst = dp()
	#dlabel, valid, centers = inst.eval(0.08, 2.5, 'nl', dataTmp)
	#inst = dpOrig()
	#dlabel, valid, centers = inst.eval(0.15, 5, 'nl', dataTmp)
	from sklearn.cluster import DBSCAN
	dbs = DBSCAN(eps=1.0, min_samples=4).fit(dataTmp)
	dlabel = dbs.labels_
	allLabel = set(dlabel)
	allLabel = list(allLabel)
	print(len(allLabel))
	num = len(allLabel)
	colors = cm.rainbow(np.linspace(0, 1, num + 1))
	dpcolorMap2 = [colors[dlabel[itm]] for itm in range(len(tmpx))]
	plt.scatter(tmpx, tmpy, c=dpcolorMap2)
	plt.figure(4)

	colorMap = {allLabel[i]: round(255/(num+2))*(i+1) for i in range(len(allLabel))}
	print(colorMap)
	for idx, data in enumerate(dataTmp):
		weak[data[0]][data[1]] = colorMap[dlabel[idx]]
	for i in range(len(weak)):
		tmp = []
		for j in range(len(weak[0])):
			weak[i][j] = 255 - weak[i][j]
			if weak[i][j] == 255: weak[i][j] = -999
	ttt = np.matrix(weak)
	masked_array=np.ma.masked_where(ttt == -999, ttt)
	cmap = cm.jet
	cmap.set_bad(color='w')
	plt.imshow(masked_array, cmap=cmap)
	plt.savefig("C:\\study\\image\\13_dpha.png",bbox_inches='tight', pad_inches = 0)
	#plt.savefig("C:\\study\\image\\13_dp.png",bbox_inches='tight', pad_inches = 0)
	plt.show()

	print('End of test!')

