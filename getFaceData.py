import os
import sys
import cv2
import torch
import torchvision.transforms.transforms as transforms
import torchvision.transforms.functional as functional
from face_detector.face_detector import DnnDetector

from SeResNeXt import se_resnext50
from utils import get_label_age
from utils import get_label_gender
from face_alignment.face_alignment import FaceAlignment
import numpy as np

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([#transforms.ToPILImage(),
                                #transforms.RandomEqualize(p=1),
                                #transforms.ColorJitter(brightness=(0.8,1.2),contrast=(0.8,1.2),saturation=(0.8,1.2),hue=(-0.05,0.05)),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.RandomRotation(degrees=10)
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
class FaceModel:
    def __init__(self):
        # AgeModel
        self.resnext = se_resnext50(num_classes=5).to(device)
        self.resnext.eval()
        # GenderModel
        self.resnextGender = se_resnext50(num_classes=2).to(device)
        self.resnextGender.eval()
        # Load Age Model
        savedModel = os.listdir("models age")[0]
        checkpoint = torch.load("models age/" + savedModel, map_location=device)
        # Load Gender Model
        savedGenderModel = os.listdir("models gender")[0]
        checkpointGender = torch.load("models gender/" + savedGenderModel, map_location=device)

        self.resnext.load_state_dict(checkpoint['resnext'])
        self.resnextGender.load_state_dict(checkpointGender['resnext'])

        self.face_alignment = FaceAlignment()

        # Face detection
        root = 'face_detector'
        self.face_detector = DnnDetector(root)

    def getFaceData(self, image, debug=True):
        
        dataDictArray = []
        # faces
        frame = image
        faces = self.face_detector.detect_faces(frame)

        for face in faces:
            # (x, y, w, h) = face

            # preprocessing
            input_face = self.face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (100, 100), interpolation=cv2.INTER_LINEAR)

            debug_img = np.array(input_face)
            if debug:
                cv2.imshow("Face" + str(len(dataDictArray)), debug_img)
            
            input_face = transform(input_face)
            
            input_face = input_face.unsqueeze(0)

            print(input_face.shape)

            with torch.no_grad():
                input_face = input_face.to(device)
                age = self.resnext(input_face)
                gender = self.resnextGender(input_face)

                torch.set_printoptions(precision=6)
                softmax = torch.nn.functional.softmax

                ages_soft = softmax(age.squeeze(), dim=-1).reshape(-1, 1).cpu().detach().numpy()
                gender_soft = softmax(gender.squeeze(), dim=-1).reshape(-1, 1).cpu().detach().numpy()

                for i, ag in enumerate(ages_soft):
                    ag = round(ag.item(), 3)

                for i, ge in enumerate(gender_soft):
                    ge = round(ge.item(), 3)

                age = torch.argmax(age)
                # percentage_age = round(ages_soft[age].item(), 2)
                age = age.squeeze().cpu().detach().item()

                gender = torch.argmax(gender)
                # percentage_gender = round(gender_soft[gender].item(), 2)
                gender = gender.squeeze().cpu().detach().item()

                dataDict = {}
                age = get_label_age(age)
                dataDict['age'] = age
                gender = get_label_gender(gender)
                dataDict['gender'] = gender
                dataDictArray.append(dataDict)

        return dataDictArray
    
    
