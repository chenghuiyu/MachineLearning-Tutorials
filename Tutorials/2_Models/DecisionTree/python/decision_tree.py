'''
主要实现决策树的C45和ID3的算法
'''

import numpy as np

class DecisionTree:

    def __init__(self, mode):
        self._tree = None
        self._mode = None

        if mode == 'ID3' or mode == 'C4.5':
            self._mode = mode
        else:
            raise Exception("mode should be id3 or c4.5")

    def _calculateEntropy(self, y_labels):
        """
        calculate entropy
        :param y_labels: 标签
        :return:
        """
        labels_num = y_labels.shape[0]
        labelCounts = {}
        for label in y_labels:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1

        entropy = 0.0
        for key in labelCounts.keys():
            pro = float(labelCounts[key])/labels_num
            entropy -= pro * np.log2(pro)
        return entropy


    def _splitDataset(self, X, y, index, value):
        ret = []
        featureValue = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]]

        for i in range(len(featureValue)):
            if featureValue[i] == value:
                ret.append(i)
        return X[ret, :], y[ret]


    def _chooseBestFeatureTosplit_ID3(self, X, y):
        """
        choose best feature to split id3
        g(D,A) = H(D) - H(Y/X) 信息增益
        :param X:
        :param y:
        :return:
        主要变量说明：
                numFeatures：特征个数
                oldEntropy：原始数据集的熵
                newEntropy：按某个特征分割数据集后的熵
                infoGain：信息增益
                bestInfoGain：记录最大的信息增益
                bestFeatureIndex：信息增益最大时，所选择的分割特征的下标
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calculateEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1

        """compute InfoGain of every feature , then choose the best one"""
        for feature in range(numFeatures):
            featureList = X[:, feature]
            uniqueValue = set(featureList)
            newEntropy = 0.0
            for value in uniqueValue:
                sub_X, sub_y = self._splitDataset(X, y, feature, value)
                pro = len(sub_y)/float(len(y))
                newEntropy += pro * self._calculateEntropy(sub_y)
            InfoGain = oldEntropy - newEntropy
            if(InfoGain > bestInfoGain):
                bestInfoGain = InfoGain
                bestFeatureIndex = feature

        return bestFeatureIndex

    def _chooseBestFeatureTosplit_C45(self, X, y):
        numFeatures = X.shape[1]
        oldEntropy = self._calculateEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        """compute InfoGain of every feature , then choose the best one"""
        for feature in range(numFeatures):
            featureList = X[:, feature]
            uniqueValue = set(featureList)
            newEntropy = 0.0
            splitInformation = 0.0
            for value in uniqueValue:
                sub_X, sub_y = self._splitDataset(X, y, feature, value)
                pro = len(sub_y)/float(len(y))
                newEntropy += pro * self._calculateEntropy(sub_y)
                splitInformation -= pro * np.log2(pro)
            if(splitInformation == 0.0):
                pass
            else:
                InfoGain = oldEntropy - newEntropy
                GainRatio = InfoGain/splitInformation
                if(GainRatio > bestInfoGain):
                    bestInfoGain = GainRatio
                    bestFeatureIndex = feature
        return bestFeatureIndex


    def _majorityCount(self, labelList):
        """
        返回labelList出现次数最多的label
        :param labelList:
        :return:
        """
        labelCount = {}
        for label in labelList:
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1
        sortClassCount = sorted(labelList.intertems(), lambda x: x[1], reverse=True)

        return sortClassCount[0][0]

    def _createTree(self, X, y, featureIndex):
        """
        build decision tree
        :param X:
        :param y:
        :param featureIndex:
        :return:
        """
        labelList = list(y)
        """same labels"""
        if(labelList.count(labelList[0]) == len(labelList)):
            return labelList[0]

        if(len(labelList) == 0):
            return self._majorityCount(labelList)

        if self._mode == 'C4.5':
            bestFeatureIndex = self._chooseBestFeatureTosplit_C45(X, y)
        elif self._mode == 'ID3':
            bestFeatureIndex = self._chooseBestFeatureTosplit_ID3(X, y)

        bestFeaStr = featureIndex[bestFeatureIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeaStr)
        featureIndex = tuple(featureIndex)

        decisionTree = {bestFeaStr:{}}
        featValues = X[:, bestFeatureIndex]
        uniqueValues = set(featValues)
        for value in uniqueValues:
            sub_X, sub_y = self._splitDataset(X, y, bestFeatureIndex, value)
            decisionTree[bestFeaStr][value] = self._createTree(sub_X, sub_y, featureIndex)
        return decisionTree


    def fit(self, X, y):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("X, y must be np.ndarry")
        featureIndex = tuple(['x' + str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X, y, featureIndex)

        return self

    def predict(self, X):
        if self._tree == None:
            raise NotFittedError("Estimator not fitted, call `fit` first")

        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("X must be np.ndarry")


        def _classify(tree, sample):
            featIndex = list(tree.keys())[0]
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])]
            valueOfKey = secondDict[key]
            if isinstance(valueOfKey, dict):
                label = _classify(valueOfKey, sample)
            else:
                label = valueOfKey
            return label

        if len(X.shape) == 1:
            return _classify(self._tree, X)
        else:
            results = []
            for i in range(X.shape[0]):
                results.append(_classify(self._tree, X[i]))

        return np.array(results)

    def show(self):
        if self._tree == None:
            raise NotFittedError("tree is None")

        import treePlotter

        treePlotter.createPlot(self._tree)

def NotFittedError(Exception):
    pass
