import torch
from ultralytics import YOLO
import cv2
import logging

# Load YOLO model with specified weights
yolo_weights_path = "C:\\luis\\videoMonitoring\\model\\YOLO\\best.pt"
yolo_model = YOLO(yolo_weights_path)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Logger configuration
logger = logging.getLogger(__name__)

# Model configuration
CONF = 0.5  # Confidence threshold for YOLO detection
IOU = 0.6   # Intersection over Union (IoU) threshold for YOLO detection

# Target class label to detect
TARGET_LABEL = "stack_light"

def YOLO_detection(frame_rgb: cv2.typing.MatLike, conf: float = CONF, iou: float = IOU, target_label: str = TARGET_LABEL) -> list[list]:
    """
    Performs YOLO object detection on the provided RGB frame and returns the bounding boxes for the specified target label.

    Parameters:
        frame_rgb (cv2.typing.MatLike): The RGB image frame to perform detection on.
        conf (float, optional): Confidence threshold for YOLO detection. Defaults to the global CONF variable.
        iou (float, optional): Intersection over Union (IoU) threshold for YOLO detection. Defaults to the global IOU variable.
        target_label (str, optional): The target class label to detect. Defaults to the global TARGET_LABEL variable.

    Returns:
        list: A list of bounding boxes where each bounding box is represented as [x1, y1, x2, y2].
              If no objects matching the target label are found, an empty list is returned.

    Logs:
        - A warning if the target label is not found in the frame.
        - An info log stating the number of objects detected that match the target label.

    Raises:
        None: The function handles errors internally by logging warnings or information messages.
    """
    # Perform inference with the YOLO model
    results = yolo_model(frame_rgb, conf=conf, iou=iou)

    # Get the class names from the YOLO model
    class_names = yolo_model.names

    label_found = False  # Flag to track if the target label is found
    
    bounding_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls.item())  # Class index of the detected object
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        
        if class_names[cls] == target_label:
            label_found = True
            bounding_boxes.append([x1, y1, x2, y2])

    if not label_found:
        bounding_boxes = [[]]
        logger.warning(f"No {target_label} found in the frame.")
    else:
        # Log the number of objects detected in the frame
        logger.info(f"Number of {target_label} found in the frame: {len(bounding_boxes)}")

    return bounding_boxes
