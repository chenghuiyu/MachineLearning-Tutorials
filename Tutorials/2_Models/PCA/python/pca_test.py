import numpy as np
from pca import PCA, LoadData
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataMat = LoadData("./data/secom.data")
    pcaMat = PCA(dataMat)
