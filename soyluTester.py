from getFaceData import FaceModel
import pyautogui
import numpy as np
import cv2
from tkinter import Tk

model = FaceModel()

image = np.array(pyautogui.screenshot())

#image = cv2.imread("woman.jpg")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


print(model.getFaceData(image, debug=True))
cv2.waitKey(0)
print(image.dtype)