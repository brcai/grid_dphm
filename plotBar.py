import numpy as np
import matplotlib.pyplot as plt

N = 6
'''
dpha = (0.666,0.762,0.79,0.927,0.996,1)
complete = (0.554,0.742,0.68,0.439,0.915,0.559)
single = (0.584,0.828,0.68,0.521,0.902,0.726)
average = (0.554,0.75,0.68,0.439,0.996,0.521)
hausdorff = (0.608, 0.654, 0.286, 0.214, 0.685, 0.477)

'''
dpha = (0.97,0.849,1,1,1,1)
complete = (0.554,0.742,0.68,0.439,0.915,0.559)
single = (0.584,0.828,0.68,0.521,0.902,0.726)
average = (0.554,0.75,0.68,0.439,0.996,0.521)
hausdorff = (0.608, 0.654, 0.286, 0.214, 0.685, 0.477)


ind = np.arange(N)  # the x locations for the groups
ind = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
width = 0.036      # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, dpha, width, color='r')
rects2 = ax.bar(ind+width, single, width, color='g')
rects3 = ax.bar(ind+width*2, complete, width, color='k')
rects4 = ax.bar(ind+width*3, average, width, color='y')
rects5 = ax.bar(ind+width*4, hausdorff, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('NMI',fontsize=15)
ax.set_title('Evaluation on different clustering distances',fontsize=15)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Pathbased', 'Compound', 'Jain', 'Flame', 'Aggregation', 'Spiral'),fontsize=15)
ax.tick_params(labelsize=15)
ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ('Proposed', 'Single', 'Complete', 'Average', 'Hausdorff'),fontsize=12)


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

plt.show()
