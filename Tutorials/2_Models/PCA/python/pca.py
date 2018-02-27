"""
Reference from the book <<Machine learning in action>>

@author: Peter Harrington
"""

import numpy as np

class PCA(object):

    def __init__(self, dataMat, percent=0.99):
        """

        :param dataMat: 输入训练集
        :param percent: 特征值数量的占比
        """
        self.percent = percent
        self.dataMat = dataMat

    def _zeroMean(self, dataMat):
        """
        对样本矩阵进行零均值处理
        :param dataMat: 按列进行求均值，每一列代表一个特征值
        :return:
        """
        meanMat = np.mean(dataMat, axis=0)
        zeroMat = dataMat - meanMat

        return zeroMat, meanMat

    def _getTopNFeatures(self, eigVals):
        """
        找到出现概率最多的N个特征
        :param eigVals: 特征值
        :return:
        """
        descendOrder = np.sort(eigVals)[-1::-1]
        orderSum = np.sum(descendOrder)
        tmp = 0
        num = 0
        for i in descendOrder:
            tmp += i
            num += 1
            if tmp > orderSum * self.percent:
                return num

    def pcaImplement(self):
        zeroMat, meanMat = self._zeroMean(self.dataMat)
        #计算协方差矩阵
        covMat = np.cov(zeroMat, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        topN = self._getTopNFeatures(eigVals)
        eigValIndice = np.argsort(eigVals)
        topN_eigValIndice = eigValIndice[-1:-(topN - 1):-1]
        topN_eigVect = eigVects[:, topN_eigValIndice]
        #构建低维的特征空间的数据
        lowDDataMat = zeroMat * topN_eigVect
        reconMat = np.multiply(lowDDataMat, topN_eigVect.T) + meanMat

        return reconMat, lowDDataMat

class LoadData(object):

    def __init__(self, filepath, delimt='\t'):
        self.filepath = filepath
        self.delimt = delimt

    def _loadDataSet(self, filepath, delimt):
        """
        加载数据
        :param filepath:文件的位置
        :param delimt:文件里面的分隔符
        :return:
        """
        file = open(filepath)
        stringArr = [line.strip().split(delimt) for line in file]
        dataArr = [map(float, str) for str in stringArr]


        return np.mat(dataArr)

    def replaceNanWithMean(self):
        """
        将数据为空的字段替换掉
        :return:
        """
        dataMat = self._loadDataSet(self.filepath, self.delimt)
        numFeat = dataMat.shape[1]
        for i in range(numFeat):
            meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
            dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal
        return dataMat

