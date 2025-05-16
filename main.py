
import argparse
import logging

import sys
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor

import cv2

from pathlib import Path
from typing import List, Union

from sdk.commons import utils
from sdk.commons.monitor import Monitor

from sdk.data.inference_source import InferenceSource

from sdk import loadmodel, loadmodelasync

from sdk.runtime.runtimeasync import RuntimeAsync
from sdk.runtime.runtime import Runtime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)
windowname = "test"

parser = argparse.ArgumentParser()
parser.add_argument("-spath", "--sample-path", type=str, default=None, help="sample path")
parser.add_argument("-smjpeg", "--sample-mjpeg", type=str, default=None, help="cctv mjpeg url")
parser.add_argument("-m", "--model-path", type=str, help="model path")
parser.add_argument("-c", "--confidence", type=int, default=50, help="confidence threshold")
parser.add_argument("-t", "--threshold", type=int, default=50, help="nms filter threshold")
parser.add_argument("-d", "--display", action="store_true", help="display inference results")
parser.add_argument("-l", "--loop", action="store_true", help="loop forever when input sample is video")
parser.add_argument("-f", "--fps", action="store_true", help="monitor inference frame per second when input sample is video")
parser.add_argument("--hailo-async", action="store_true", help="startup hailo module in async mode")

args = parser.parse_args()

sample_path  = args.sample_path # "/home/avalue/hailosdk/samples/images"
sample_mjpeg = args.sample_mjpeg

model_path = args.model_path #"/home/avalue/hailosdk/models/object-detection/yolo/onnx/yolo11x.onnx"

confidence = args.confidence
threshold = args.threshold

is_display = args.display
is_loop = args.loop
is_monitor = args.fps

is_async = args.hailo_async

logger.debug(f"is_display: {is_display}, is_loop: {is_loop}, is_monitor: {is_monitor}")
logger.debug(f"samples: {sample_path} {sample_mjpeg}")

monitor = Monitor(Path(model_path).name)

def main():
    if (is_display):
        cv2.namedWindow(
            windowname,
            cv2.WINDOW_NORMAL
        )
        
        cv2.setWindowProperty(
            windowname,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN, 
        )
        
    if (is_async):
        runtime = loadmodelasync(model_path)
    else:
        runtime = loadmodel(model_path)
    
    if (runtime is None):
        logger.error("runtime is undefined")
        sys.exit()
        
    monitor.get_temperature = lambda : runtime.temperature
    monitor.get_information = lambda : runtime.information
    
    if (is_monitor):
        monitor.start()
    
    if (sample_path is not None):
        path = Path(sample_path)
        if(path.is_file() or path.is_dir()):
            sample_files = utils.fileslist(sample_path)
            display_inference_files(runtime, sample_files)
            
    elif (sample_mjpeg is not None):
        if(sample_mjpeg.find("http") >= 0):
            display_inference_url_mjpeg(runtime, sample_mjpeg)
        
    if (is_display):
        cv2.destroyAllWindows()
        
def display_inference_files(
    runtime: Union[Runtime, RuntimeAsync],
    sample_files: List[str]
):
    index = 0
    max = len(sample_files)
    
    while True:
        sample_file = sample_files[index]
        
        logger.info(f"start up with {sample_file} at {index}")
        
        is_image, is_video = utils.filextension(sample_file)
    
        if (is_image):
            logger.info(f"inference image: {sample_file}")
            display_inference_image(runtime, sample_file)
    
        elif (is_video):
            if (is_async and isinstance(runtime, RuntimeAsync)):
                logger.info(f"inference async video: {sample_file}")
                display_inference_video_async(runtime, sample_file)
                
            elif (not is_async and isinstance(runtime, Runtime)):
                logger.info(f"inference video: {sample_file}")
                display_inference_video(runtime, sample_file)
            
        else:
            logger.error(f"sample_file: {sample_file} both not image or video")
        
        
        if (is_video):
            if (is_loop):
                
                if (max > 1):
                    index = (index + 1) % max
                    logger.info(f"next video again")
                
                key = cv2.waitKeyEx(1)
                
            else:
                logger.info(f"wait for key, q = exit, left/right arrow to control prev/next sample files")
                key = cv2.waitKeyEx(0)
                
        else:
            key = cv2.waitKeyEx(0)
        
        if key == ord('q') or key == ord('Q'):
            break

        elif key == 65361:  # Left Arrow
            index = (index - 1 + max) % max
            
        elif key == 65363:  # Right Arrow
            index = (index + 1) % max
            
def display_inference_image(runtime: Runtime, filepath: str) -> None:
    logger.info(filepath)
    
    input_image = utils.read_image(filepath)
    
    result = runtime.inference(
        InferenceSource(
            input_image,
            confidence,
            threshold,
            time.time()
        )
    )
    
    if (is_display):
        cv2.imshow(windowname, result.image)

def display_inference_video(runtime: Runtime, filepath: str) -> None:
    capture = utils.read_video(filepath)
    
    while capture.isOpened():
        
        ret, frame = capture.read()
        if (not ret):
            break
        
        result = runtime.inference(
            InferenceSource(
                frame,
                confidence,
                threshold,
                time.time()
            )
        )
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)
            utils.drawmodelname(result.image, runtime.information.name)
            utils.drawspendtime(result.image, monitor.spandtime)
            utils.drawfps(result.image, monitor.framecount)
        
        if (is_display):
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break

def display_inference_url_mjpeg(runtime: Runtime, url: str) -> None:
    capture = utils.read_url_video(url)
    
    if (not capture.isOpened()):
        logger.error(f"open url failed url: {url}")
        return None
    
    while(True):
        
        ret, frame = capture.read()
        if (not ret):
            capture = utils.read_url_video(url)
            continue
        
        result = runtime.inference(
            InferenceSource(
                frame,
                confidence,
                threshold,
                time.time()
            )
        )
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)
            utils.drawmodelname(result.image, runtime.information.name)
            utils.drawspendtime(result.image, monitor.spandtime)
            utils.drawfps(result.image, monitor.framecount)
        
        if (is_display):
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
           
def display_inference_video_async(runtime: RuntimeAsync, filepath: str) -> None:
    
    f_feed = True
    
    pool = ThreadPoolExecutor()
    
    def __task_put_source(frame: cv2.typing.MatLike):
        runtime.put(
            InferenceSource(
                frame,
                confidence,
                threshold,
                time.time()
            )
        )
    
    def __task_feed_video():
        capture = utils.read_video(filepath)
        
        while capture.isOpened() and f_feed:
            ret, frame = capture.read()
            if (not ret):
                break
            
            while not runtime.avaliable():
                time.sleep(0.001)
                continue
            
            pool.submit(__task_put_source, frame)
                
            runtime.put(
                InferenceSource(
                    frame,
                    confidence,
                    threshold,
                    time.time()
                )
            )

        capture.release()

    t_feed = threading.Thread(target=__task_feed_video, daemon=True)
    t_feed.start()    
    
    t_run = threading.Thread(target=runtime.start, daemon=True)
    t_run.start()
    
    logger.info(f"wait for first frame ")
    while runtime.get() is None:
        time.sleep(0.001)
        continue
    
    logger.info(f"start showing inference results")
    while True:
        result = runtime.get()
        
        # logger.debug(f"runtime.get: {result}")
        if result is None:
            # logger.debug(f"result is None")
            time.sleep(0.001)
            continue
        
        if (is_monitor):
            monitor.add_count()
            monitor.add_spendtime(result.spendtime)

            if (is_display):
                utils.drawmodelname(result.image, runtime.information.name)
                utils.drawspendtime(result.image, monitor.spandtime)
                utils.drawfps(result.image, monitor.framecount)
        
        if (is_display):
            
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
            
        if (not t_feed.is_alive()):
            break
          
    
    # if (is_display):
    #     while True:
    #         result = runtime.get()
    #         if (result is None):
    #             break
            
    #         cv2.imshow(windowname, result.image)
    
    logger.info(f"wait for task ...")
    f_feed = False
    runtime.stop()
    pool.shutdown(False)
    t_run.join(1)
    t_feed.join(1)
    logger.info(f"wait for task ok")

if __name__ == "__main__":
    main()
