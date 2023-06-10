import torch
import torchvision.transforms as transforms
from SeResNeXt import se_resnext50
from PIL import Image

# Load the model
model = se_resnext50(num_classes=5)
checkpoint = torch.load("saved models/resnext_37_dataset_age_UTK_custom_64_0.005_40_1e-06.pth.tar")
model.load_state_dict(checkpoint['resnext'])
model.eval()

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the image
image_path = 'data/dataset_age_UTK_custom/Test/5_Old/11.jpg'
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class label
_, predicted_idx = torch.max(output, 1)
predicted_label = predicted_idx.item()

print("Predicted class label:", predicted_label)