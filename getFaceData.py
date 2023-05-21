import sys
import cv2
import torch
import torchvision.transforms.transforms as transforms
import torchvision.transforms.functional as functional
from face_detector.face_detector import DnnDetector

from SeResNeXt import se_resnext50
from utils import get_label_age
from face_alignment.face_alignment import FaceAlignment

sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceModel:
    def __init__(self):
        # Model
        self.resnext = se_resnext50(num_classes=5).to(device)
        self.resnext.eval()
        # Load model
        checkpoint = torch.load("checkpoint/y_35_dataset_age_64_0.05_40_1e-06.pth.tar", map_location=device)
        self.resnext.load_state_dict(checkpoint['resnext'])

        self.face_alignment = FaceAlignment()

        # Face detection
        root = 'face_detector'
        self.face_detector = DnnDetector(root)


    def getFaceData(self, image):
        ageList = []
        # faces
        frame = image
        faces = self.face_detector.detect_faces(frame)

        for face in faces:
            #(x, y, w, h) = face

            # preprocessing
            input_face = self.face_alignment.frontalize_face(face, frame)
            input_face = cv2.resize(input_face, (100, 100))

            input_face = transforms.ToTensor()(input_face).to(device)

            input_face = functional.convert_image_dtype(input_face, dtype=torch.uint8)
            input_face = functional.equalize(input_face)
            input_face = functional.convert_image_dtype(input_face, dtype=torch.float32)

            input_face = torch.unsqueeze(input_face, 0)
            
            

            with torch.no_grad():
                input_face = input_face.to(device)
                age = self.resnext(input_face)

                torch.set_printoptions(precision=6)
                softmax = torch.nn.functional.softmax

                ages_soft = softmax(age.squeeze(), dim=-1).reshape(-1, 1).cpu().detach().numpy()

                for i, ag in enumerate(ages_soft):
                    ag = round(ag.item(), 3)

                age = torch.argmax(age)
                #percentage_age = round(ages_soft[age].item(), 2)
                age = age.squeeze().cpu().detach().item()

                age = get_label_age(age)
                ageList.append(age)
        return ageList