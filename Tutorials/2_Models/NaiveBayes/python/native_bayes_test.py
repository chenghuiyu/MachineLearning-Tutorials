import numpy as np
from native_bayes import MultiNB, GaussianNB

if __name__ == '__main__':
    X = np.array([
                      [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                      [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
                              ])
    X = X.T
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    result = MultiNB(alpha=1.0, fit_prior=True)
    result.fit(X, y)
    print(result.alpha)
    print(result.class_prior)
    print(result.classes)
    print(result.conditional_prob)
    print(result.predict(np.array([2, 4])))

    result2 = GaussianNB(alpha=0.0)
    print(result2.fit(X, y).predict(X))
