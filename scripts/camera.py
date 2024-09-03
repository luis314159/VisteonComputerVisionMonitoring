import cv2
import logging
import time
#logger
logger = logging.getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def get_camera(camera_index: int, time_sleep :int=5):
    cap = cv2.VideoCapture(camera_index)
    try:
        # Espera un pequeño tiempo para permitir que la cámara se inicie
        time.sleep(time_sleep)  # Puedes ajustar este tiempo según sea necesario
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

def list_cameras(max_cameras: int = 10) -> list:
    """
    Lists the indices of available cameras.

    Parameters:
        max_cameras (int, optional): The maximum number of camera indices to check.
                                     Defaults to 10.

    Returns:
        list: A list of available camera indices.
    """
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras