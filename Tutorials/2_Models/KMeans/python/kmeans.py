
import numpy as np

class Kmeans(object):

    def __init__(self, n_cluster=5, initCent='random', max_iter=300):
        """

        :param n_cluster: 聚类的个数
        :param initCent:  质心初始化
        :param max_iter:  最大迭代次数
        """
        if getattr(initCent, '__array__'):
            n_cluster = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None

        self.n_cluster = n_cluster
        self.initCent = initCent
        self.max_iter = max_iter
        self.culsterAssmen = None
        self.sse = None
        self.labels = None

    def _distEuclidean(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def _randomCent(self, X_input, k):
        #n_dimen表示输入样本的特征维数
        n_dimen = X_input.shape[1]
        centroids = np.empty((k, n_dimen))

        for i in range(n_dimen):
            min_value = np.min(X_input[:, i], keepdims=True)
            range_value = float(np.max(X_input[:, i], keepdims=True) - min_value)
            centroids[:, i] = (min_value + range_value * np.random.rand(k, 1)).flatten()
        return centroids


    def fit(self, X_input):
        if not isinstance(X_input, np.ndarray):
            try:
                X_input = np.array(X_input)
            except:
                raise Exception("numpy.ndarray required for X_input")
        samplesNum = X_input.shape[0]
        # culsterAssmen: matrix (m * 2) ,
        # first column is number of cluster,
        # second column is RMS
        self.culsterAssmen = np.empty((samplesNum, 2))

        if self.initCent == 'random':
            self.centroids = self._randomCent(X_input, self.n_cluster)

        clusterChanged = True
        for _ in range(self.max_iter):
            for i in range(samplesNum):
                minDist = np.inf
                minIndex = -1
                for j in range(self.n_cluster):
                    # centroids: K * feature_dimns
                    distJI = self._distEuclidean(self.centroids[j, :], X_input[i, :])
                    if(distJI < minDist):
                        minDist = distJI
                        minIndex = j
                if(self.culsterAssmen[i, 0] != minIndex):
                    clusterChanged = True
                    self.culsterAssmen[i, :] = minIndex, minDist**2

            if not clusterChanged:
                break

            for n in range(self.n_cluster):
                ptsInCluster = X_input[np.nonzero(self.culsterAssmen[:, 0] == n)[0]]
                self.centroids[n, :] = np.mean(ptsInCluster, axis=0)

        self.labels = self.culsterAssmen[:, 0]
        self.sse = np.sum(self.culsterAssmen[:, 1])

    def predict(self, X_input):
        if not isinstance(X_input, np.ndarray):
            try:
                X_input = np.array(X_input)
            except:
                raise Exception("numpy.ndarray required for X_input")

        # numbers of samples
        sampleNums = X_input.shape[0]
        preds = np.empty((sampleNums, ))
        for n in range(sampleNums):
            minDist = np.inf
            for m in range(self.n_cluster):
                distJI = self._distEuclidean(self.centroids[m, :], X_input[n, :])
                if(distJI < minDist):
                    minDist = distJI
                    preds[m] = n
        return preds

