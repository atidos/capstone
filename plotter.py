import time

import matplotlib.pyplot as plt
import numpy as np
import random

def get_train_data(log):
    with open(log, "r") as f:
        list = f.read().split("\n")

    epochList = []
    train_loss = []
    val_loss = []
    Accuracy = []
    Percision = []
    Recall = []

    for string in list:
        if (string != ""):
            epochList.append(string)

    count = 0
    for x in epochList:
        index = epochList.index(x)
        if index % 3 == 0:
            count += 1
            # print(x.split("loss=")[1])
            train_loss.append(float(x.split("loss=")[1]))
        elif index % 3 == 1:
            # print(x)
            val_loss.append(float(x.split("loss=")[1]))
        else:
            Accuracy.append(float(x.split(" ")[2]))
            Percision.append(float(x.split(" ")[7]))
            Recall.append(float(x.split(" ")[12]))

    arr = np.array(Recall)

    data = {
        "epochs": range(0, count),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": Accuracy,
        "percision": Percision,
        "recall": Recall
    }

    return data

def plotAccuracy(data, labels=[], colors=[]):

    for i in range(len(data)):
        datum = data[i]
        label = labels[i] if len(labels) > i else None
        color = colors[i] if len(colors) > i else None
        plt.plot(datum["accuracy"], label=label, color=color)

    plt.grid()
    plt.legend()
    plt.ylim(0,100)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def plotLoss(data, labels=[], colors=[]):

    for i in range(len(data)):
        datum = data[i]
        label = labels[i] if len(labels) > i else None
        color = colors[i] if len(colors) > i else None
        plt.plot(datum["val_loss"], label=label, color=color)

    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.show()

plt.title("resnext50_dataset_age_64_0.005_40_1e-06")

#a = get_train_data("logs/resnext50_dataset_age_UTK_64_0.005_40_1e-06")
#plotLoss([a])

X = ["batch size = 64\nlr = 0.005",
     "batch size = 64\nlr = 0.0001",
     "batch size = 64\nlr = 0.0005",
     "batch size = 128\nlr = 0.0005",
]

X_axis = np.arange(len(X))

Y_acc = [76.4,76.8,77.3,78.1]
Y_val = [0.55,0.58,0.564,0.584]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(X_axis - 0.2, Y_acc, 0.4, color="b", label = 'Accuracy')
ax2.bar(X_axis + 0.2, Y_val, 0.4, color="r", label = 'Validation Loss')
plt.xticks(X_axis, X)
plt.title("Comparison of model's final iterations")
ax1.set_ylabel('Accuracy', color='b')
ax2.set_ylabel('Validation Loss', color='r')
ax1.set_ylim([74,79])
ax2.set_ylim([0.5,0.6])
plt.show()