
import argparse
import logging

import sys
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor

import cv2

from pathlib import Path
from typing import List, Tuple, Union

import numpy

from sdk.commons import utils
from sdk.commons.monitor import Monitor

from sdk.data.inference_result import InferenceResult
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
parser.add_argument("--no-inference", action="store_true", help="no inference consumer")
parser.add_argument("--sample-random", action="store_true", help="auto generate random buffer array as inference source")
parser.add_argument("--sample-random-width", type=int, default=1920, help="with --sample-random buffer width size")
parser.add_argument("--sample-random-height", type=int, default=1080, help="with --sample-random buffer height size")

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

no_inference = args.no_inference

is_sample_random = args.sample_random
sample_random_size = (args.sample_random_width, args.sample_random_height)

logger.debug(f"is_display: {is_display}, is_loop: {is_loop}, is_monitor: {is_monitor}")
logger.debug(f"samples: {sample_path} {sample_mjpeg}")


def main():
    monitor = Monitor(Path(model_path).name)
    monitor.start()
    
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
        runtime = loadmodelasync(monitor, model_path)
    else:
        runtime = loadmodel(monitor, model_path)
    
    if (runtime is None):
        logger.error("runtime is undefined")
        sys.exit()
    
    runtime.display = True if is_display else False
    
    if (sample_path is not None):
        path = Path(sample_path)
        if(path.is_file() or path.is_dir()):
            sample_files = utils.fileslist(sample_path)
            display_inference_files(runtime, sample_files)
            
    elif (sample_mjpeg is not None):
        if(sample_mjpeg.find("http") >= 0):
            display_inference_url_mjpeg(runtime, sample_mjpeg)
            
    elif (is_sample_random):
        
        if (is_async and isinstance(runtime, RuntimeAsync)):
            display_inference_random_async(runtime, sample_random_size)
            
        else:
            display_inference_random(runtime, sample_random_size)
        
    else:
        logger.error("inference sample resource is undefined")
        sys.exit()
        
    if (is_display):
        cv2.destroyAllWindows()
        
def display_monitor(image: cv2.typing.MatLike, runtime: Union[Runtime, RuntimeAsync]):
    utils.drawmodelname(image=image, name=runtime.information.name)
    utils.drawlatency(image=image, spendtime=runtime.latency)
    utils.drawfps(image=image, framecount=runtime.fps)
    utils.drawsize(image=image, width=image.shape[1], height=image.shape[0])
        
def display_inference_files(
    runtime: Union[Runtime, RuntimeAsync],
    sample_files: List[str]
):
    index = 0
    max = len(sample_files)
    
    while True:
        
        try:
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

            runtime.reset()
            
            if (is_video):
                if (is_loop):
                    
                    index = (index + 1) % max
                    logger.info(f"next video")
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
        
        except Exception as e:
            logger.error(e)
            
def display_inference_random(runtime: Runtime, size: Tuple[int, int]) -> None:
    logger.info("start inference random buffer array")
    
    while True:
        
        buffer = numpy.random.rand(size[1], size[0], 3) * 255
        frame = buffer.astype(numpy.uint8)
        
        result = (
            InferenceResult(
                InferenceSource(
                    frame,
                    0,
                    0,
                    time.time()
                )
            )
            if no_inference
            else
            runtime.inference(
                InferenceSource(
                    frame,
                    confidence,
                    threshold,
                    time.time()
                )
            )
        )
        
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (is_display):
            display_monitor(result.image, runtime)
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
            
def display_inference_random_async(runtime: RuntimeAsync, size: Tuple[int, int]) -> None:
    logger.info("start inference random buffer array")
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
        while f_feed:
            buffer = numpy.random.rand(size[1], size[0], 3) * 255
            frame = buffer.astype(numpy.uint8)
            
            while not runtime.avaliable():
                continue
            
            try:
                pool.submit(__task_put_source, frame)
                
            except Exception as e:
                logger.error(e)

    t_feed = threading.Thread(target=__task_feed_video, daemon=True)
    t_feed.start()    
    
    t_run = threading.Thread(target=runtime.run, daemon=True)
    t_run.start()
    
    logger.info(f"wait for first frame ")
    while runtime.get() is None:
        time.sleep(0.001)
        continue
    
    logger.info(f"start showing inference results")
    rterr = 0
    while rterr < 10000:
        result = runtime.get()
        
        # logger.debug(f"runtime.get: {result}")
        if result is None:
            # logger.debug(f"result is None")
            time.sleep(0.001)
            rterr += 1
            continue
        
        rterr = 0
                
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (is_display):
            display_monitor(result.image, runtime)
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
            
        if (not t_feed.is_alive()):
            break
             
    logger.info(f"wait for task terminate ...")
    f_feed = False
    runtime.stop()
    pool.shutdown(False)
    t_run.join(1)
    t_feed.join(1)
    logger.info(f"wait for task terminate done")

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
        
        result = (
            InferenceResult(
                InferenceSource(
                    frame,
                    0,
                    0,
                    time.time()
                )
            )
            if no_inference
            else 
            runtime.inference(
                InferenceSource(
                    frame,
                    confidence,
                    threshold,
                    time.time()
                )
            )
        )
        
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (is_display):
            display_monitor(result.image, runtime)
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
        
        result = (
            InferenceResult(
                InferenceSource(
                    frame,
                    0,
                    0,
                    time.time()
                )
            )
            if no_inference
            else 
            runtime.inference(
                InferenceSource(
                    frame,
                    confidence,
                    threshold,
                    time.time()
                )
            )
        )
        
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (is_display):
            display_monitor(result.image, runtime)
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
                continue

            try:
                pool.submit(__task_put_source, frame)
                
            except Exception as e:
                logger.error(e)

        capture.release()

    t_feed = threading.Thread(target=__task_feed_video, daemon=True)
    t_feed.start()    
    
    t_run = threading.Thread(target=runtime.run, daemon=True)
    t_run.start()
    
    logger.info(f"wait for first frame ")
    while runtime.get() is None:
        time.sleep(0.001)
        continue
    
    logger.info(f"start showing inference results")
    rterr = 0
    while rterr < 10000:
        result = runtime.get()
        
        # logger.debug(f"runtime.get: {result}")
        if result is None:
            # logger.debug(f"result is None")
            time.sleep(0.001)
            rterr += 1
            continue
        
        rterr = 0
                
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (is_display):
            display_monitor(result.image, runtime)
            cv2.imshow(windowname, result.image)
            key = cv2.waitKeyEx(1)
            if key == ord('q') or key == ord('Q'):
                break
            
        if (not t_feed.is_alive()):
            break
          
    logger.info(f"wait for task terminate ...")
    f_feed = False
    runtime.stop()
    pool.shutdown(False)
    t_run.join(1)
    t_feed.join(1)
    logger.info(f"wait for task terminate done")

if __name__ == "__main__":
    main()
