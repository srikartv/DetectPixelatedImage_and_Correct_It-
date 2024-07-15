import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def restore_image(model, image_path, transform, device):
    # Load and preprocess the image
    bimg = cv2.imread(image_path)
    bimg = transform(bimg)
    bimg = bimg.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        corrected_image = model(bimg)
    
    # Convert the output tensor to an image
    corrected_image = corrected_image.squeeze(0)  # Remove batch dimension
    corrected_image = corrected_image.cpu().numpy()
    corrected_image = np.transpose(corrected_image, (1, 2, 0))  # CHW to HWC
    return corrected_image

if __name__ == '__main__':
    # Load the trained model
    model_path = 'unet_model.pth'
    model = load_model(model_path, device)

    # Provide the path to the blurred image
    blurred_image_path = 'E:/Intel/new approach/dataset/train/pixel/7_NIKON-D3400-35MM_S.JPG'

    # Prepare transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    # Restore the image
    restored_image = restore_image(model, blurred_image_path, transform, device)

    # Display the restored image
    plt.imshow(restored_image)
    plt.axis('off')
    plt.show()
