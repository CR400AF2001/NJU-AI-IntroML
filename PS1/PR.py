import pandas
import matplotlib.pyplot as plt


data = pandas.read_csv("data.csv").sort_values(by = "output", ascending = False)
preList = []
recList = []
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
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    preList.append(p)
    recList.append(r)
plt.title('P-R Curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.plot(recList, preList)
plt.show()