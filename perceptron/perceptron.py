import numpy as np
import time



def loadData(fileName):
    '''
    load MNIST dataset
    :param fileName: dataset path
    :return dataArr: dataset list
    :return labelArr: label list
    '''
    print('start to load data')
    dataArr, labelArr = [], []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # dichotomy, divide into >= 5 && < 5 group
        if (int(curLine[0])) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        # uniform before add into data
        dataArr.append([int(num)/255 for num in curLine[1:]])
    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter=50):
    '''
    train perceptron
    :param dataArr: list of data
    :param labelArr: list of label
    :param iter: training iteration
    :return: w&b: weights and biases
    '''
    # transform data into np matrix for computing convenience
    # column vectors in default
    print('start transforming')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T # transpose for format uniformity
    m, n = np.shape(dataMat)
    w = np.zeros((1, dataMat.shape[1])) # weight
    b = 0 # bias
    h = 1e-3 # learning rate

    # iterations
    for k in range(iter):
        # apply gradient descent for each sample
        # how to apply stochastic gradient descent?
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            # update params if misclassification
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi
        print(f'Round {k + 1}/{iter} training')
    return w, b

def model_test(dataArr, labelArr, w, b):
    '''
    test perceptron
    :param dataArr: list of data
    :param labelArr: list of label
    :param w & b: trained weights and bias of perceptron
    :return accuRate: model accuracy
    '''
    print('start testing')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T + b)
        if result >= 0: errorCount += 1
    accuRate = 1 - (errorCount / m)
    return accuRate

if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')
    w, b = perceptron(trainData, trainLabel)
    accuRate = model_test(testData, testLabel, w, b)
    end = time.time()
    print("Time span: {:.3f} sec\t Accuracy: {:.2f}%".format(end - start, accuRate * 100))