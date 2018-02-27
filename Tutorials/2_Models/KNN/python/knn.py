"""
implete knn
"""

import numpy as np
import operator

def classifier(X_input, dataSet, labels, k):
    dataSet_size = dataSet.shape[0]
    diffDataSet = np.tile(X_input, [dataSet_size, 1]) - dataSet
    diffMat = np.power(diffDataSet, 2)
    diffDataDistance = np.sum(diffMat, axis=1)
    diffDataSq = np.sqrt(diffDataDistance)
    sortedDistIndics = diffDataSq.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndics[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = np.zeros([1, 1024])
    file = open(filename)
    for i in range(32):
        listfile = file.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = listfile[j]
    return returnVect