import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jpype import *
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

def clustering(open_prices,close_prices):
    # The daily variations of the quotes are what carry most information
    variation = close_prices - open_prices

    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphicalLassoCV(cv=5)

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    # Cluster using affinity propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    clusters = []
    for i in range(n_labels + 1):
        clusters.append(symbols[labels == i])
        print('Cluster %i: %s' % ((i + 1), ', '.join(clusters[i])))

    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane
    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense', n_neighbors=6)
    embedding = node_position_model.fit_transform(X.T).T

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
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels, cmap=plt.cm.nipy_spectral)

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
             embedding[0].max() + .10 * embedding[0].ptp())
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
    plt.show()

    return clusters



def calculateTE(arrayList):
    result = []
    l = list(itertools.permutations(arrayList, 2))
    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
    for array in l:
        teCalc.initialise(1, 0.5) # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        teCalc.setObservations(JArray(JDouble, 1)(array[0][0]), JArray(JDouble, 1)(array[1][0]))
        result.append((teCalc.computeAverageLocalOfObservations(),array[0][1]+"->"+array[1][1]))
    return result



symbols = np.array(["AMZN","AAPL","GOOG","FB","NFLX","V",
"NVDA","AMD",
"MSFT","IBM","INTC","TSM","CSCO",
"BA","LMT","NOC","RTN","UTX",
"GS","JPM","C","BAC","AXP","DIS",
"JNJ","MCD","UNH","WMT","WBA",
"CVX","XOM","BP",
"KO","VZ","T"])
quotes = [pd.read_csv(s+".csv") for s in symbols]
close_prices = np.vstack([q['Close'] for q in quotes])
open_prices = np.vstack([q['Open'] for q in quotes])
clusters = clustering(open_prices,close_prices)

startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=./infodynamics.jar")
prices = {symbols[i]:(quotes[i]["Close"].values.tolist()) for i in range(len(symbols))}


for c in clusters:
    arrayList = [(prices[i],i) for i in c]
    res = calculateTE(arrayList)
    print(c)
    print(res)


