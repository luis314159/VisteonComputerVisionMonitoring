import torch
from ultralytics import YOLO
import cv2

yolo_weights_path = "C:\luis\videoMonitoring\model\YOLO\best.pt"
yolo_model = YOLO(yolo_weights_path)

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

CONF = 0.5
IOU = 0.6
TARGET_LABEL = "stack_light"

def plot_frame_with_classification(frame):

    global IOU, CONF, TARGET_LABEL
    # Convertir el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hacer la inferencia con el modelo YOLO
    results = yolo_model(frame_rgb, conf=CONF, iou=IOU)
    
    # Obtener los nombres de las clases del modelo YOLO
    class_names = yolo_model.names

    label_found = False  # Variable para rastrear si se encontró el target_label
    
    bounding_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        if class_names[cls] == target_label:
            label_found = True  # Se encontró el target_label

            # Recortar la imagen en la región del bounding box
            cropped_img = frame_rgb[y1:y2, x1:x2]
            
            # Pasar la imagen recortada al modelo de clasificación
            predicted_class = classification_model(cropped_img)
            
            # Añadir la etiqueta de la clasificación a la imagen original
            label = f'{predicted_class}'
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Guardar la información del bounding box
            bounding_boxes.append(
                [x1, y1, x2, y2]
            )

        else:
            # Dibujar el bounding box con la etiqueta original
            original_label = class_names[cls]
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_rgb, original_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not label_found:
        print(f"No {target_label} found in the frame.")
    
    # Convertir de vuelta a BGR para mostrar usando OpenCV o guardar la imagen
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    return frame_bgr, bounding_boxes