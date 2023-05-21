from getFaceData import FaceModel
import cv2

model = FaceModel()
print(model.getFaceData(cv2.imread("soylu.jpg")))

