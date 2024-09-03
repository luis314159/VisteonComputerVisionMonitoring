from YOLO_object_detection import YOLO_detection
import logging
import cv2
from camera import capture_image
from aux_modules import save_bounding_boxes, plot_frame, disable_logging, plot_images, crop_img
from shufflenet import classificate_state, shufflenet_transform

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def main():
    frame = capture_image()
    #convert to RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bounding_boxes = YOLO_detection(frame_rgb)
    save_bounding_boxes(bounding_boxes)
    #disable_logging(plot_frame)(frame_rgb)
    cropped_images = crop_img(frame_rgb, bounding_boxes, shufflenet_transform)
    #disable_logging(plot_images)(cropped_images)
    labels = classificate_state(cropped_images)
    
    logging.debug(f"Labels: {labels}")

    # cap = cv2.VideoCapture(0)
    # # Set mediapipe model 
    # while cap.isOpened():
            
    #     cropped_images = crop_img(frame_rgb, bounding_boxes, shufflenet_transform)
    #     disable_logging(plot_images)(cropped_images)
    #     labels = classificate_state(cropped_images)
        
    #     logging.debug(f"Labels: {labels}")

    #     # Break gracefully
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()