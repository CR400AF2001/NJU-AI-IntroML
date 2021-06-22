from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True)
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)
print(trainx.shape, testx.shape, trainy.shape, testy.shape)

# linear regression
trainx = np.mat(trainx)
trainy = np.mat(trainy).T
testx = np.mat(testx)
testy = np.mat(testy).T
m = trainx.shape[0]
d = trainx.shape[1]
v1 = np.mat([1] * m).T
w = np.linalg.inv(trainx.T * trainx - trainx.T * v1 * v1.T * trainx / m) * \
    (trainx.T * trainy - trainx.T * v1 * v1.T * trainy / m)
b = (trainy.T * v1 - (trainy.T * trainx - trainy.T * v1 * v1.T * trainx / m) *
     np.linalg.inv(trainx.T * trainx - trainx.T * v1 * v1.T * trainx / m) * (trainx.T * v1)) / m
mse = 0
for i in range(0, testx.shape[0]):
    mse += (float(testy[i]) - float(testx[i] * w + b)) ** 2
mse /= testx.shape[0]
print(mse)
# ridge regression
lambdaList = []
trainmseList = []
testmseList = []
for i in range(10, 10000):
    lambdaValue = i / 10
    lambdaList.append(lambdaValue)
    w = np.linalg.inv(trainx.T * trainx + 2 * lambdaValue * np.identity(d) - trainx.T *
        v1 * v1.T * trainx / m) * (trainx.T * trainy - trainx.T * v1 * v1.T * trainy / m)
    b = (trainy.T * v1 - (trainy.T * trainx - trainy.T * v1 * v1.T * trainx / m) * np.linalg.inv(trainx.T *
        trainx + 2 * lambdaValue * np.identity(d) - trainx.T * v1 * v1.T * trainx / m) * (trainx.T * v1)) / m
    mse = 0
    for i in range(0, trainx.shape[0]):
        mse += (float(trainy[i]) - float(trainx[i] * w + b)) ** 2
    mse /= trainx.shape[0]
    trainmseList.append(mse)
    mse = 0
    for i in range(0, testx.shape[0]):
        mse += (float(testy[i]) - float(testx[i] * w + b)) ** 2
    mse /= testx.shape[0]
    testmseList.append(mse)
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.plot(lambdaList, trainmseList)
plt.plot(lambdaList, testmseList)
plt.show()
