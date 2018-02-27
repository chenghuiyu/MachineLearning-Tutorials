"""
对实现的函数进行单元测试
"""

from decision_tree import DecisionTree


if __name__ == '__main__':
    # Toy data
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']

    clf = DecisionTree(mode='ID3')
    clf.fit(X, y)
    clf.show()
    print
    clf.predict(X)  # ['yes' 'yes' 'no' 'no' 'no']

    clf_ = DecisionTree(mode='C4.5')
    clf_.fit(X, y).show()
    print
    clf_.predict(X)  # ['yes' 'yes' 'no' 'no' 'no']
