import pandas
import matplotlib.pyplot as plt


data = pandas.read_csv("data.csv").sort_values(by = "output", ascending = False)
tprList = []
fprList = []
length = len(data)
tp = fp = tn = fn = 0
for i in data["label"]:
    if i == 0:
        tn += 1
fn = length - tn
for i in data["label"]:
    if i == 1:
        tp += 1
        fn -= 1
    else:
        fp += 1
        tn -= 1
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    tprList.append(tpr)
    fprList.append(fpr)
plt.title('ROC Curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.plot(fprList, tprList)
plt.show()


listLength = len(fprList)
auc = 0
for i in range(0, listLength - 1):
    auc += (fprList[i + 1] - fprList[i]) * (tprList[i] + tprList[i + 1])
auc /= 2
print(auc)