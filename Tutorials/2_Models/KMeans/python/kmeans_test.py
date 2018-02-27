
from six.moves import cPickle
import matplotlib.pyplot as plt
import numpy as np
from kmeans import Kmeans

if __name__ == '__main__':
    with open('data.pkl', 'rb') as plkFile:
        x_train, y_train = cPickle.load(plkFile, encoding='iso-8859-1')

    max_inters = 10

    for max_inter in range(max_inters):
        n_cluster = 5
        initCent = x_train[50: 60]
        clf = Kmeans(n_cluster, initCent, max_inter)
        clf.fit(x_train)
        cents = clf.centroids
        labels = clf.labels
        sse = clf.sse
        # drow the images of results
        colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
        for i in range(n_cluster):
            index = np.nonzero(labels == i)[0]
            x0 = x_train[index, 0]
            x1 = x_train[index, 1]
            y_i = y_train[index]
            for j in range(len(x0)):
                plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], fontdict={'weight': 'bold', 'size': 9})
            plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)

        plt.title("SSE={:.2f}".format(sse))
        plt.axis([-30, 30, -30, 30])
        plt.show()

