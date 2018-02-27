"""
主要实现贝叶斯估计，定义类MultiNB，并利用高斯估计的方式实现GaussianNB
"""



import numpy as np


class MultiNB(object):
    """
    Naive Bayes classifier for multinomial models
        The multinomial Naive Bayes classifier is suitable for classification with
        discrete features
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """

        :param alpha: 贝叶斯估计的参数，alpha=1时，就是拉普拉斯平滑估计
        :param fit_prior: 是否需要计算Y的前向概率，如果设为false，那么就会有一个默认的值，即满足最大熵的均值等概论分布
        :param class_prior:Y的前向概率分布
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        # 条件概率：P(X=x_i | y =c_k)
        self.conditional_prob = None

    def _caculate_feature_prob(self, features):
        """
        计算估计特征值的前向概率
        :param feature: 样本特征
        :return:
        """
        feature_values = np.unique(features)
        featureNum = len(features)
        features_prob = {}
        for value in feature_values:
            feature_sum = np.sum(np.equal(value, features))
            features_prob[value] = \
                ((feature_sum + self.alpha)) / (featureNum + len(feature_values) * self.alpha)

        return features_prob


    def fit(self, X_train, y_labels):
        """
        对样本数据进行训练
        :param X_train: 训练样本集
        :param y_labels: 训练标签
        :return:
        """

        if not isinstance(X_train, np.ndarray):
            try:
                X_train = np.array(X_train)
            except:
                raise Exception("Input should be type of array")

        self.classes = np.unique(y_labels)
        if self.class_prior == None:
            classNum = len(self.classes)
            if not self.fit_prior:
                self.class_prior = ((1 / classNum) for _ in range(classNum))
            else:
                self.class_prior = []
                labelsNum = float(len(y_labels))
                for c in self.classes:
                    c_numbers = np.sum(np.equal(c, y_labels))
                    self.class_prior.append((c_numbers + self.alpha) / (labelsNum + classNum * self.alpha))

        # 条件概率 P(X = x_i | Y = c_k)
        # 字典类型类似于 { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(X_train.shape[1]):
                feature = X_train[np.equal(c, y_labels)][:, i]
                self.conditional_prob[c][i] = self._caculate_feature_prob(feature)
        return self


    def _get_xj_prob(self, values_prob, target_value):
        return values_prob[target_value]


    def _predict_single_sample(self, x):
        label = -1
        max_posterior_prob = 0

        for c_index in self.classes:
            current_class_prior = self.class_prior[c_index]
            current_condition_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_id in feature_prob.keys():
                current_condition_prob = self._get_xj_prob(feature_prob[feature_id], x[j])
                j += 1
            if current_condition_prob * current_class_prior > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_condition_prob
                label = self.classes[c_index]
        return label

    def predict(self, X):

        if X.ndim == 1:
            return self._predict_single_sample(X)

        else:
            labels = []
            for i in range(X.shape[0]):
                label = self._predict_single_sample(X[i])
                labels.append(label)
            return labels


class GaussianNB(MultiNB):
    """
    使用高斯概率函数进行贝叶斯参数估计
    """

    def _caculate_feature_prob(self, features):
        """
        计算特征的概率密度
        :param features:
        :return:
        """
        mu = np.mean(features)
        sigma = np.std(features)

        return (mu, sigma)

    def _prob_gauss(self, mu, sigma, x):
        pro_power = -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))
        pro_va = 1 / (sigma * np.sqrt(2 * np.pi))
        return pro_va * np.exp(pro_power)

    def _get_xj_prob(self, mu_sigma, target_value):
        return self._prob_gauss(mu_sigma[0], mu_sigma[1], target_value)
