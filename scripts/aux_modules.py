import json
import logging
from datetime import datetime
import os
from typing import List
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

BOUNDING_BOXES_FILE_PATH = "C:\\luis\\videoMonitoring\\logs\\bounding_boxes.json"

# Logger configuration
logger = logging.getLogger(__name__)

def save_bounding_boxes(bounding_boxes: List[List], file_path: str = BOUNDING_BOXES_FILE_PATH) -> None:
    """
    Save a single entry for a group of bounding boxes to a JSON file with a maximum limit of 1000 entries.
    
    All bounding boxes in the provided list are recorded together with a single timestamp representing 
    the time of capture. If the total number of entries exceeds 1000, the oldest entries will be discarded 
    to maintain the limit.
    
    Parameters:
    ----------
    bounding_boxes : List[List]
        A list where each element is a list representing a bounding box. Each bounding box 
        should contain information such as [x_min, y_min, x_max, y_max].
    
    file_path : str, optional
        The path to the JSON file where the bounding boxes should be saved. The default is 'bounding_boxes.json'.
    
    Returns:
    -------
    None
    """
    try:
        # Load existing bounding boxes from the file if it exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    logging.warning(f"{file_path} is empty or contains invalid JSON. Initializing a new list.")
                    data = []
        else:
            logging.warning(f"{file_path} does not exist. Creating a new file.")
            data = []

        # Add a single entry for the group of bounding boxes with a common timestamp
        timestamp = datetime.now().isoformat()
        data.append({'timestamp': timestamp, 'bounding_boxes': bounding_boxes})

        # Ensure the list doesn't exceed 1000 entries
        if len(data) > 1000:
            data = data[-1000:]  # Keep only the last 1000 entries

        # Save the updated bounding boxes back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        logging.info(f'Successfully saved {len(bounding_boxes)} bounding boxes as a single entry to {file_path}')
    except Exception as e:
        logging.error(f'Failed to save bounding boxes: {str(e)}')

def plot_frame(frame:cv2.typing.MatLike):
    from matplotlib import pyplot as plt
    plt.imshow(frame)
    plt.show()

def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Revert the normalization applied to a tensor image for visualization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

def plot_images(images: torch.Tensor, cols: int = 3) -> None:
    """
    Display all images in the provided tensor in a single plot using Matplotlib.
    
    Parameters:
    ----------
    images : torch.Tensor
        A tensor of images with shape (batch_size, channels, height, width).
    
    cols : int, optional
        The number of columns in the grid. The default is 3.
    
    Returns:
    -------
    None
    """
    from matplotlib import pyplot as plt
    num_images = images.size(0)  # Get the batch size

    if num_images < cols:
        cols = num_images

    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * rows))  # Adjust figure size based on the number of rows
    
    # Unnormalize images before plotting
    images = unnormalize(images)

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        
        # Permute the dimensions from (C, H, W) to (H, W, C)
        img = images[i].permute(1, 2, 0).cpu().numpy()

        plt.imshow(np.clip(img, 0, 1))  # Ensure values are in [0, 1] range for displaying
        plt.title(f'Image {i+1}')
        plt.axis('off')  # Hide axes for better visualization
    
    plt.tight_layout()
    plt.show()


def disable_logging(func):
    def wrapper(*args, **kwargs):
        # Guardar el nivel de logging actual
        previous_level = logging.getLogger().getEffectiveLevel()
        
        # Establecer un nivel alto para ignorar logs
        logging.getLogger().setLevel(logging.CRITICAL)
        
        try:
            return func(*args, **kwargs)
        finally:
            # Restaurar el nivel de logging original
            logging.getLogger().setLevel(previous_level)
    
    return wrapper

def crop_img(image: cv2.Mat, bounding_boxes: List[List[int]], transform) -> torch.Tensor:
    """
    Crop the given image based on the provided bounding boxes and apply the specified transformation
    pipeline to each cropped image. The transformed images are then stacked into a single PyTorch tensor.
    
    Parameters:
    ----------
    image : cv2.Mat
        The original image RGB from which the crops will be made.
    
    bounding_boxes : List[List[int]]
        A list of bounding boxes where each bounding box is represented as [x1, y1, x2, y2].
    
    transform : callable
        The transformation pipeline to apply to each cropped image. This should include resizing, 
        conversion to tensor, and normalization as needed.
    
    Returns:
    -------
    torch.Tensor
        A tensor containing all transformed and cropped images stacked along the batch dimension.
    """
    from matplotlib import pyplot as plt
    cropped_images = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cropped_img = image[y1:y2, x1:x2]

        # Convert the cropped image (numpy array) to PIL Image
        pil_img = Image.fromarray(cropped_img)
        pil_img = pil_img.convert('RGB')
        # Apply the transformation pipeline
        transformed_img = transform(pil_img)
        
        cropped_images.append(transformed_img)
    return torch.stack(cropped_images)  # Stack all tensors to create a batch

