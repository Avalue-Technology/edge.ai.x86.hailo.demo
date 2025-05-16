import asyncio
import logging
from multiprocessing import Process
import threading
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)

from sdk.commons import utils
from sdk.runtime.hailortasync import HailortAsync

if __name__ == "__main__":

    rt = HailortAsync(
        "sdk/models/object-detection/yolo/hailo-8-hef/yolov11n.hef",
        False,
        10,
        10
    )
    video = "sdk/samples/videos/20200229174849.mp4"

    captrue = utils.read_video(video)

    t_run = threading.Thread(target=rt.run, daemon=True)
    # threading.Thread(target=rt.start, daemon=True).start()
    # asyncio.run(rt.run())
    # t_run = asyncio.create_task(rt.run())

    t_run.start()

    while captrue.isOpened():
        
        ret, frame = captrue.read()
        if(not ret):
            break
        
        now = time.time()
        
        # logger.debug(f"read frame {now}")
        while not rt.avaliable:
            time.sleep(0.001)
        
        rt.put(
            frame,
            10,
            10,
            now,
        )
        
        time.sleep(0.001)
        
    t_run.join()
        