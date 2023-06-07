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

a = get_train_data("logs/resnext50_dataset_age_UTK_64_0.005_40_1e-06")
plotLoss([a])

