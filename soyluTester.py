from getFaceData import FaceModel
import cv2

model = FaceModel()
model.getFaceData(cv2.imread("soylu.jpg"))

