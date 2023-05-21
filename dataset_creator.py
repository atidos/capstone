import cv2
import matplotlib.pyplot as plt

import os

# assign directory
directory = 'origin/manual'

count = 0
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        file1 = open(f, 'r')
        lines = file1.readlines()

        fileName = filename
        category = int(lines[7])
        categoryName = ""

        if category == 0:
            categoryName = "0-3"
        if category == 1:
            categoryName = "4-19"
        if category == 2:
            categoryName = "20-39"
        if category == 3:
            categoryName = "40-69"
        if category == 4:
            categoryName = "70+"

        subDataset = fileName.split("_")[0].title()
        dataName = fileName.split("_")
        img = plt.imread("origin/aligned/" + fileName.split("_")[0] + "_" + fileName.split("_")[1] + "_aligned.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        try:
            #img = cv2.resize(img, dsize=(48, 48))
            cv2.imwrite("dataset_age/" + subDataset + "/" + categoryName + "/" + str(count) + ".jpg", img)
            #img = cv2.flip(img, 1)
            #cv2.imwrite("dataset2/" + subDataset + "/" + categoryName + "/" + str(count) + "_f.jpg", img)
            if count % 1000 == 0:
                print(count)
            count += 1

        except Exception as e:
            ...
