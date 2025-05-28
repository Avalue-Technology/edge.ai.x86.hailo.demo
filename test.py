
import logging
import time

import cv2

from frame.random import Random


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)

windowname = __name__


cv2.namedWindow(
    windowname,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    windowname,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_GUI_EXPANDED, #cv2.WINDOW_FULLSCREEN, 
)


# c = VideoCapture("/home/avalue/hailodemo/sdk/samples/videos/20200229174849.mp4")
c = Random(640, 640)
c.start()

while(True):
    
    frame = c.get()
    if (frame is None):
        time.sleep(0.001)
        continue
        
    # cv2.imshow(windowname, frame)
    # cv2.waitKeyEx(delay=1)

cv2.destroyAllWindows()