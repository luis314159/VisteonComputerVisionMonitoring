import cv2
import logging
#logger
logger = logging.getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def get_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            logger.warning("Failed to access the camera.")
            yield None
        else:
            yield cap
    finally:
        cap.release()

def capture_image(camera_index: int = 0) -> cv2.typing.MatLike:
    """
    Captures a single image frame from the specified camera.

    Parameters:
        camera_index (int, optional): The index of the camera to access.
                                     Defaults to 0.

    Returns:
        cv2.typing.MatLike: The captured image frame if successful.
                             Returns None if the camera cannot be accessed
                             or if the frame capture fails.
    """
    with get_camera(camera_index) as cap:
        if cap is None:
            return None
        ret, frame = cap.read()
        if ret:
            return frame
        else:
            logger.warning("Failed to capture the image.")
            return None
