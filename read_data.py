import re

def scalarTmp(data):
	for i in range(len(data[0])):
		maxVal = -10000.
		dataTmp = data.copy()
		for j in range(len(data)):
			if data[j][i] > maxVal: maxVal = data[j][i]
		for j in range(len(data)):
			if maxVal != 0: dataTmp[j][i] = data[j][i]/maxVal
	return dataTmp


class readDataFile:
	def iris():
		fp = open('C:/study/real data/iris/iris.data.txt')
		data = []
		label = []
		labelMap = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(labelMap[tmp[-1]])
		return data, label

	def wine():
		fp = open('C:/study/real data/wine/wine.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def heart():
		fp = open('C:/study/real data/heart/heart.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def wdbc():
		fp = open('C:/study/real data/wdbc/wdbc.data.txt')
		data = []
		label = []
		labelMap = {'M':0, 'B':1}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[2:]]
			data.append(vec)
			label.append(labelMap[tmp[1]])
		return data, label

	def waveform():
		fp = open('C:/study/real data/waveform/waveform.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def ionoshpere():
		fp = open('C:/study/real data/ionosphere/ionosphere.data.txt')
		data = []
		label = []
		labelMap = {'g':0 ,'b':1}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(labelMap[tmp[-1]])
		return data, label

	def pendigits():
		fp = open('C:/study/real data/pendigits/pendigits.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def japanesev():
		print("Not valid")
		exit(0)
		fp = open('C:/study/real data/japanesev/pendigits.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def monk3():
		fp = open('C:/study/real data/monk3/monk3.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[2:-1]]
			data.append(vec)
			label.append(int(tmp[1]))
		return data, label

	def movement_libras():
		fp = open('C:/study/real data/movement_libras/movement_libras.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def semeion():
		fp = open('C:/study/real data/semeion/semeion.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			newTmp = [itm for itm in tmp if itm != '']
			if len(newTmp) <= 2: continue
			vec = [float(itm) for itm in newTmp[0:-10]]
			data.append(vec)
			tmpL = [int(i) for i in newTmp[-10:]]
			label.append(tmpL.index(1))
		return data, label

	def spambase():
		fp = open('C:/study/real data/spambase/spambase.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def wholesale():
		fp = open('C:/study/real data/wholesale/wholesale.csv')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			if i==0: i += 1; continue
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def zoo():
		fp = open('C:/study/real data/zoo/zoo.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def spect():
		fp = open('C:/study/real data/spect/spect.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def pima_india_diabetes():
		fp = open('C:/study/real data/pima_india_diabetes/pima-indians-diabetes.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def bcw():
		fp = open('C:/study/real data/bcw/bcw.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def bupa():
		fp = open('C:/study/real data/bupa/bupa.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def ecoli():
		fp = open('C:/study/real data/ecoli/ecoli.data.txt')
		data = []
		label = []
		labelMap = {'cp':0, 'im':1, 'imS':2, 'imU':3, 'imL':4, 'om':5, 'omL':6, 'pp':7}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1] if itm != '']
			data.append(vec)
			label.append(labelMap[tmp[-1]])
		return data, label

	def musk():
		fp = open('C:/study/real data/musk/musk.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[2:-1]]
			data.append(vec)
			label.append(float(tmp[-1]))
		return data, label

	def pageb():
		fp = open('C:/study/real data/pageb/pageb.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(float(tmp[-1]))
		return data, label

	def seeds():
		fp = open('C:/study/real data/seeds/seeds.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split('\t')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(float(tmp[-1]))
		return data, label

	def sonar():
		fp = open('C:/study/real data/sonar/sonar.data.txt')
		data = []
		label = []
		labelMap = {'R':0, 'M':1}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(labelMap[tmp[-1]])
		return data, label

	def haber():
		fp = open('C:/study/real data/haber/haber.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def abalone():
		fp = open('C:/study/real data/abalone/abalone.data.txt')
		data = []
		label = []
		labelMap = {'M':1, 'F':2, 'I':3}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(labelMap[tmp[0]])
		return data, label

	def balance():
		fp = open('C:/study/real data/balance/balance.data.txt')
		data = []
		label = []
		labelMap = {'B':1, 'R':2, 'L':3}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(labelMap[tmp[0]])
		return data, label

	def cmc():
		fp = open('C:/study/real data/cmc/cmc.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def glass():
		fp = open('C:/study/real data/glass/glass.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def hayes():
		fp = open('C:/study/real data/hayes/hayes.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def lenses():
		fp = open('C:/study/real data/lenses/lenses.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def lung():
		fp = open('C:/study/real data/lung/lung.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def tae():
		fp = open('C:/study/real data/tae/tae.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def australian():
		fp = open('C:/study/real data/australian/australian.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('?', '1')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def plrx():
		fp = open('C:/study/real data/plrx/plrx.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('\t', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp if itm != '']
			data.append(vec[0:-1])
			label.append(int(vec[-1]))
		return data, label

	def fert():
		fp = open('C:/study/real data/fertility/fertility.data.txt')
		data = []
		label = []
		lmap = {'N':0, 'O':1}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace('\t', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec[0:-1])
			label.append(lmap[tmp[-1]])
		return data, label

	def knowledge():
		fp = open('C:/study/real data/knowledge/knowledge.data.txt')
		data = []
		label = []
		lmap = {'very_low':0, 'High':1, 'Low':2, 'Middle':3}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split('\t')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(lmap[tmp[-1]])
		return data, label

	def skillcraft():
		fp = open('C:/study/real data/skillcraft/skillcraft.csv')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def wilt():
		fp = open('C:/study/real data/wilt/wilt.data.txt')
		data = []
		label = []
		i = 0
		lmap = {'w':0, 'n':1}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split('\t')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:] if itm != '']
			data.append(vec)
			label.append(lmap[tmp[0]])
		return data, label

	def epileptic():
		fp = open('C:/study/real data/epileptic/data.csv')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def page():
		fp = open('C:/study/real data/page/page.data.txt')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def eeg():
		fp = open('C:/study/real data/eeg/eeg.data.txt')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def bank():
		fp = open('C:/study/real data/bank/bank.data.txt')
		data = []
		label = []
		i = 0
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def frog():
		fp = open('C:/study/real data/frog/frog.data.txt')
		data = []
		label = []
		lmap = {'Leptodactylidae':0, 'Dendrobatidae':1, 'Hylidae':3, 'Bufonidae':4}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split('\t')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(lmap[tmp[-1]])
		return data, label

	def dermatology():
		fp = open('C:/study/real data/dermatology/dermatology.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-2]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def soybean():
		fp = open('C:/study/real data/soybean/soybean.data.txt')
		data = []
		label = []
		lmap = {'D1':1,'D2':2,'D3':3,'D4':4}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(lmap[tmp[-1]])
		return data, label

	def segment():
		fp = open('C:/study/real data/segment/segment.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def shuttle():
		fp = open('C:/study/real data/shuttle/shuttle.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def thyroid():
		fp = open('C:/study/real data/thyroid/thyroid.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def vowel():
		fp = open('C:/study/real data/vowel/vowel.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[3:] if itm != '']
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def magic():
		fp = open('C:/study/real data/magic/magic.data.txt')
		data = []
		label = []
		lmap = {'g':1, 'h':2}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(lmap[tmp[-1]])
		return data, label

	def transfusion():
		fp = open('C:/study/real data/transfusion/transfusion.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = line.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def winequality():
		fp = open('C:/study/real data/winequality/winequality.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(';')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def winequalityw():
		fp = open('C:/study/real data/winequalityw/winequalityw.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(';')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label

	def parkinsons():
		fp = open('C:/study/real data/parkinsons/parkinsons.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[1:]]
			data.append(vec)
			label.append(int(tmp[0]))
		return data, label

	def column():
		fp = open('C:/study/real data/column/column.data.txt')
		data = []
		label = []
		lmap = {'AB':1, 'NO':2}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split(' ')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1] if itm != '']
			data.append(vec)
			label.append(lmap[tmp[-1]])
		return data, label

	def indian():
		fp = open('C:/study/real data/indian/indian.csv')
		data = []
		label = []
		lmap = {'Female':1 ,'Male':2}
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.replace(' ', '')
			tmp = tmp.split(',')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[2:]]
			data.append(vec)
			label.append(lmap[tmp[1]])
		return data, label

	def skin():
		fp = open('C:/study/real data/skin/skin.data.txt')
		data = []
		label = []
		for line in fp.readlines():
			tmp = line.replace('\n', '')
			tmp = tmp.split('\t')
			if len(tmp) <= 2: continue
			vec = [float(itm) for itm in tmp[0:-1]]
			data.append(vec)
			label.append(int(tmp[-1]))
		return data, label



def loadData(dataSet):
	data = []
	label = []
	if dataSet == 'iris':
		data, label = readDataFile.iris()
	elif dataSet == 'wine':
		data, label = readDataFile.wine()
	elif dataSet == 'heart':
		data, label = readDataFile.heart()
	elif dataSet == 'wdbc':
		data, label = readDataFile.wdbc()
	elif dataSet == 'ionosphere':
		data, label = readDataFile.ionoshpere()
	elif dataSet == 'waveform':
		data, label = readDataFile.waveform()
	elif dataSet == 'pendigits':
		data, label = readDataFile.pendigits()
	elif dataSet == 'japanesev':
		data, label = readDataFile.japanesev()
	elif dataSet == 'monk3':
		data, label = readDataFile.monk3()
	elif dataSet == 'movement':
		data, label = readDataFile.movement_libras()
	elif dataSet == 'semeion':
		data, label = readDataFile.semeion()
	elif dataSet == 'spambase':
		data, label = readDataFile.spambase()
	elif dataSet == 'wholesale':
		data, label = readDataFile.wholesale()
	elif dataSet == 'zoo':
		data, label = readDataFile.zoo()
	elif dataSet == 'spect':
		data, label = readDataFile.spect()
	elif dataSet == 'pima':
		data, label = readDataFile.pima_india_diabetes()
	elif dataSet == 'seeds':
		data, label = readDataFile.seeds()
	elif dataSet == 'bcw':
		data, label = readDataFile.bcw()
	elif dataSet == 'sonar':
		data, label = readDataFile.sonar()
	elif dataSet == 'pageb':
		data, label = readDataFile.pageb()
	elif dataSet == 'bupa':
		data, label = readDataFile.bupa()
	elif dataSet == 'ecoli':
		data, label = readDataFile.ecoli()
	elif dataSet == 'musk':
		data, label = readDataFile.musk()
	elif dataSet == 'haber':
		data, label = readDataFile.haber()
	elif dataSet == 'abalone':
		data, label = readDataFile.abalone()
	elif dataSet == 'balance':
		data, label = readDataFile.balance()
	elif dataSet == 'cmc':
		data, label = readDataFile.cmc()
	elif dataSet == 'glass':
		data, label = readDataFile.glass()
	elif dataSet == 'hayes':
		data, label = readDataFile.hayes()
	elif dataSet == 'lenses':
		data, label = readDataFile.lenses()
	elif dataSet == 'lung':
		data, label = readDataFile.lung()
	elif dataSet == 'tae':
		data, label = readDataFile.tae()
	elif dataSet == 'australian':
		data, label = readDataFile.australian()
	elif dataSet == 'plrx':
		data, label = readDataFile.plrx()
	elif dataSet == 'fert':
		data, label = readDataFile.fert()
	elif dataSet == 'knowledge':
		data, label = readDataFile.knowledge()
	elif dataSet == 'skillcraft':
		data, label = readDataFile.skillcraft()
	elif dataSet == 'wilt':
		data, label = readDataFile.wilt()
	elif dataSet == 'epileptic':
		data, label = readDataFile.epileptic()
	elif dataSet == 'page':
		data, label = readDataFile.page()
	elif dataSet == 'eeg':
		data, label = readDataFile.eeg()
	elif dataSet == 'bank':
		data, label = readDataFile.bank()
	elif dataSet == 'frog':
		data, label = readDataFile.frog()
	elif dataSet == 'dermatology':
		data, label = readDataFile.dermatology()
	elif dataSet == 'soybean':
		data, label = readDataFile.soybean()
	elif dataSet == 'segment':
		data, label = readDataFile.segment()
	elif dataSet == 'shuttle':
		data, label = readDataFile.shuttle()
	elif dataSet == 'thyroid':
		data, label = readDataFile.thyroid()
	elif dataSet == 'vowel':
		data, label = readDataFile.vowel()
	elif dataSet == 'magic':
		data, label = readDataFile.magic()
	elif dataSet == 'transfusion':
		data, label = readDataFile.transfusion()
	elif dataSet == 'winequality':
		data, label = readDataFile.winequality()
	elif dataSet == 'winequalityw':
		data, label = readDataFile.winequalityw()
	elif dataSet == 'parkinsons':
		data, label = readDataFile.parkinsons()
	elif dataSet == 'column':
		data, label = readDataFile.column()
	elif dataSet == 'indian':
		data, label = readDataFile.indian()
	elif dataSet == 'skin':
		data, label = readDataFile.skin()
	else: 
		print('no such dataset')
		exit(0)
	
	dataTmp = scalarTmp(data)
	return dataTmp, label

if __name__ == '__main__':
	print('Test start')
	#readDataFile.iris()
	#readDataFile.wine()
	#readDataFile.heart()
	#readDataFile.wdbc()
	#readDataFile.ionoshpere()
	#readDataFile.waveform()
	#readDataFile.pendigits()
	#readDataFile.bcw()
	#readDataFile.bupa()
	#readDataFile.ecoli()
	#readDataFile.monk3()
	#readDataFile.musk()
	#readDataFile.pageb()
	#readDataFile.pima_india_diabetes()
	#readDataFile.seeds()
	#readDataFile.sonar()
	#readDataFile.spect()
	readDataFile.wholesale()
	print('The end')