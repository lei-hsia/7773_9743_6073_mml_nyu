
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd
import pandas_datareader as pdr

from sklearn import cluster, covariance, manifold

整个项目的主要过程: 
    1. retreive data from Yahoo Finance
    2. 根据有的data得到数据点集之间的相关性,更specifically,得到correlations矩阵:有了所有的correlation之后,就相当于learned a graphical
        structure from correlations; (其中correlation由covariance.GraphicalLassoCV得到), 得到是一个edge_model;
    3. 有了这个model/correlation之后就cluster这些数据点;用cluster.AffinityPropagation;
    4. cluster完成之后就想visualize; 为了最佳的visualization效果, i.e. 找到每个stock node的最佳位置, 用manifold.LocallyLinearEmbedding;
    5. 找好了之后就真正开始plot;
    6. 用jpype包中的东西, 创建计算TE的计算器，然后开始算, 最终得到的是TE的矩阵, 表示的是每个stock price对其他的price产生的影响
    

# #############################################################################
# Retrieve the data from Internet

# The data is from 2003 - 2008. This is reasonably calm: (not too long ago so
# that we get high-tech firms, and before the 2008 crash). This kind of
# historical data can be obtained for from APIs like the quandl.com and
# alphavantage.co ones.
start_date = "2013-01-01"
end_date = "2017-12-31"

symbols = ["AMZN","AAPL","MSFT","GOOG","FB","NFLX",
"IBM","INTC","NVDA","AMD","TSM","CSCO",
"BA","LMT","NOC","RTN",
"GS","JPM","C","BAC","AXP","V",
"JNJ","MCD","UTX","UNH","CVX","KO","DIS","XOM","VZ","WMT","WBA","T"]
symbols = np.array(symbols)

quotes = []

for symbol in symbols:
    print('Fetching quote history for %r' % symbol, file=sys.stderr)
    quotes.append(pdr.get_data_yahoo(symbol,start=start_date,end=end_date))

close_prices = np.vstack([q['Close'] for q in quotes])
open_prices = np.vstack([q['Open'] for q in quotes])

# The daily variations of the quotes are what carry most information
variation = close_prices - open_prices


# #############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphicalLassoCV(cv=5)

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

# #############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

print(labels)

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(symbols[labels == i])))

# #############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# #############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.nipy_spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(symbols, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()

##################################


from jpype import *
import random
import math
import itertools

# Create a TE calculator and run it:
def teCal(sourceArray, destArray):
    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
    teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
    teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))
    # For copied source, should give something close to 1 bit:
    result = teCalc.computeAverageLocalOfObservations()
    print("TE result %.4f " % (result))


# Change location of jar to match yours:
jarLocation = "./infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

arr_list = []
for i in quotes:
    arr_list.append(i["Close"].tolist())

# get every pair from those stocks
teArray = list(itertools.combinations(arr_list, 2))

for i in teArray:
    teCal(i[0], i[1])  # 1st & 2nd from each pair

    
--------------------------------------------------------------

里面用到了这句: 
teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel

用到了```infodynamics.measures.continuous.kernel```包中的```TransferEntropyCalculatorKernel.java```
而这个包头文件中的import: 


import infodynamics.measures.continuous.TransferEntropyCalculator;          168 lines
import infodynamics.measures.continuous.TransferEntropyCommon;              365 lines
import infodynamics.measures.continuous.kernel.TransferEntropyKernelCounts; 48  lines
import infodynamics.utils.MathsUtils;                                       669 lines
import infodynamics.utils.MatrixUtils;                                      4895 lines
import infodynamics.utils.EmpiricalMeasurementDistribution;                 132 lines 
import infodynamics.utils.RandomGenerator;                                  842 lines


一共 7119 lines;

整个java项目(JDIT):  https://github.com/jlizier/jidt/tree/master/java




