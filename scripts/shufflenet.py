import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from PIL import Image

SHUFFLENET_PATH = "C://luis//videoMonitoring//model//shufflenet//model7.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(pretrained=True, fine_tune=True):
    """
    Builds and returns a ShuffleNet v2 x2.0 model with options to load pretrained weights 
    and fine-tune the model.

    Parameters:
    ----------
    pretrained : bool, optional
        If True, loads the model with pretrained weights. Defaults to True.
    
    fine_tune : bool, optional
        If True, all layers are set to require gradients (i.e., the model will be fine-tuned).
        If False, all layers except the final fully connected layer are frozen. Defaults to True.
    
    Returns:
    -------
    model : nn.Module
        A ShuffleNet v2 x2.0 model with the final fully connected layer adapted to 5 output classes.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    
    model = models.shufflenet_v2_x2_0(pretrained=pretrained)
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    else:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification layer to 5 classes
    model.fc = nn.Linear(2048, 5)
    return model

# Image transformation pipeline
shufflenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Optionally, you could center crop instead of resizing
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dictionary to map predicted class indices to class names
class_names_dict = {
    0: 'Blue',
    1: 'Green',
    2: 'None',
    3: 'Red',
    4: 'Yellow'
}

# Load and prepare the ShuffleNet model
shufflenet = build_model(pretrained=False, fine_tune=True)
shufflenet.load_state_dict(torch.load(SHUFFLENET_PATH))
shufflenet.eval()
shufflenet.to(device)

def classificate_state(batch_cropped_imgs):
    """
    Classifies the state of a batch of cropped images using the loaded ShuffleNet model.

    Parameters:
    ----------
    batch_cropped_imgs : list of numpy.ndarray or list of PIL.Image.Image
        The list of cropped images to classify. Each image in the batch should be either a NumPy array or a PIL image.
    
    Returns:
    -------
    list of str
        A list of class names predicted for each image in the batch.
    """
    # Ensure all input images are PIL images, converting from NumPy arrays if necessary
    #batch_cropped_imgs = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in batch_cropped_imgs]

    #batch_cropped_imgs = torch.stack(batch_cropped_imgs)
    # Pass the batch of images through the classification model
    with torch.no_grad():
        batch_tensor = batch_cropped_imgs.to(device)  # Ensure the batch tensor is on the correct device
        outputs = shufflenet(batch_tensor)
        predicted_class_idxs = torch.argmax(outputs, dim=1)
    
    # Map the predicted class indices to class names
    predicted_classes = [class_names_dict.get(idx.item(), 'Unknown') for idx in predicted_class_idxs]

    return predicted_classes


