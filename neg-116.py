import re
import numpy
from density_peak_clustering import dp
from read_data import scalarTmp
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
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

	dir = 1
	while True:
		print(dir)
		image = read_pgm("C:\\Users\\bcai\\Downloads\\CarData.tar\\CarData\\TrainImages\\neg-"+str(dir)+".pgm", byteorder='<')
		plt.figure(1)
		plt.imshow(image, cm.gray)
		plt.show()
		dir += 1

	image = read_pgm("C:\\Users\\bcai\\Downloads\\CarData.tar\\CarData\\TrainImages\\neg-116.pgm", byteorder='<')
	for ldx, line in enumerate(image):
		tmp = []
		for idx, itm in enumerate(line):
			if itm < 80:
				tmp.append(255)
			else:
				tmp.append(0)
		weak.append(tmp)
	plt.figure(1)
	plt.imshow(image, cm.gray)
	plt.figure(2)
	plt.imshow(weak, cm.gray)
	#plt.show()
	plt.figure(3)
	dataTmp = []
	for i in range(len(weak)):
		tmp = []
		for j in range(len(weak[0])):
			if weak[i][j] != 0: tmp = [i,j]; dataTmp.append(tmp)
	inst = dp()
	tmpx = [data[0] for data in dataTmp]
	tmpy = [data[1] for data in dataTmp]
	dlabel, valid, centers = inst.eval(0.06, 2, 'nl', dataTmp)
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
	plt.imshow(weak, cm.gray)
	plt.show()

	print('End of test!')

