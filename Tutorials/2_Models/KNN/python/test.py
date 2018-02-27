from os import listdir
from knn import img2vector
from knn import classifier
import numpy as np

if __name__ == '__main__':
    print("Begins\n")
    hwLabels = []
    trainFileList = listdir("trainingDigits")
    m = len(trainFileList)
    trainMat = np.zeros([m, 1024])
    for i in range(m):
        fileNameList = trainFileList[i]  # 1_0.txt
        fileList = fileNameList.split('.')[0]
        classList = int(fileList.split('_')[0])
        hwLabels.append(classList)
        trainMat[i, :] = img2vector('trainingDigits/%s' % fileNameList)

    testFileList = listdir("testDigits")
    errorCount = 0.0
    lenTest = len(testFileList)
    testMat = np.zeros([lenTest, 1024])
    for j in range(lenTest):
        testNameList = testFileList[j]
        testList = testNameList.split('.')[0]
        testClass = int(testList.split('_')[0])
        vectorTest = img2vector('testDigits/%s' % testNameList)
        classResult = classifier(vectorTest, trainMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classResult, testClass))

        if (classResult != testClass):
            errorCount += 1

        print("\nthe total number of errors is: %d" % errorCount)

        print("\nthe total error rate is: %f" % (errorCount / float(vectorTest)))

    print("Done\n")