import cv2
import torch
from torchvision.transforms import transforms
from SeResNeXt import se_resnext50
from utils import get_label_age, get_label_gender
from face_alignment.face_alignment import FaceAlignment
from face_detector.face_detector import DnnDetector

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = se_resnext50(num_classes=5)
model.load_state_dict(torch.load('models age/resnext_37_dataset_age_UTK_custom_64_0.005_40_1e-06.pth.tar')["resnext"])
model.eval()
model.to(device)

# Initialize face alignment and face detection
face_alignment = FaceAlignment()
face_detector = DnnDetector('face_detector')

# Set transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_detector.detect_faces(frame)

    for face in faces:
        # Preprocess face image
        #input_face = face_alignment.frontalize_face(face, frame)
        try:
            input_face = frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            print(0)
            
        except:
            print(1)
            input_face = face_alignment.frontalize_face(face, frame)

        input_face = cv2.resize(input_face, (100, 100))
        input_face = preprocess(input_face).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            input_face = input_face.to(device)
            age_logits = model(input_face)

        # Post-processing
        age = torch.argmax(age_logits).item()
        age_label = get_label_age(age)
        
        # Draw bounding box and text on the frame
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()