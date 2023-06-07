from getFaceData import FaceModel
import pyautogui
import numpy as np
import cv2
from tkinter import Tk

model = FaceModel()

#image = np.array(pyautogui.screenshot())
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.imread("data/dataset_gender_UTK/Test/2_Female/757.jpg")
image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_LINEAR)
print(model.getFaceData(image, debug=True))
cv2.waitKey(0)
print(image.dtype)