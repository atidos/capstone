from getFaceData import FaceModel
import pyautogui
import numpy as np
import cv2
from tkinter import Tk

model = FaceModel()

image = pyautogui.screenshot()
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
print(model.getFaceData(image))

print(image.dtype)