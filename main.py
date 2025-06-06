
import argparse
import logging
import sys
import threading
import time

from pathlib import Path
from typing import Tuple, Union

import cv2

import gc 
import tracemalloc
import objgraph

import guppy
from pympler import summary, muppy, asizeof, tracker
from sdk.data.circular_buffer import CircularBuffer

from arguments import Arguments
from frame.sample import Sample

from sdk.commons import utils
from sdk.commons.monitor import Monitor
from sdk.data.inference_result import InferenceResult
from sdk.data.inference_source import InferenceSource
from sdk.runtime.runtime import Runtime
from sdk.runtime.runtimeasync import RuntimeAsync
from sdk import loadmodel

from frame.streaming import Streaming
from frame.random import Random


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)

KEY_q = ord('q')
KEY_Q = ord('Q')
KEY_ARROW_LEFT = 65361
KEY_ARROW_RIGHT = 65363

# gc.set_debug(gc.DEBUG_LEAK)



def window_init(name: str):
    cv2.namedWindow(
        name,
        cv2.WINDOW_NORMAL
    )
    
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FREERATIO, # cv2.WINDOW_FULLSCREEN, 
    )
    
def window_close():
    cv2.destroyAllWindows()
    
def window_waitkey(delay: int):
    # ord('q')
    # key 65361 left arrow
    # key 65363 right arrow
    return cv2.waitKeyEx(delay)

def window_show(name: str, frame: cv2.typing.MatLike):
    cv2.imshow(name, frame)

def initruntime(args: Arguments) -> Union[Runtime, RuntimeAsync]:
    runtime = loadmodel(args)
    if (isinstance(runtime, RuntimeAsync)):
        runtime.size = args.streaming_size
            
    return runtime

def initstreaming(args: Arguments) -> Streaming:
    streaming = None
    if (args.sample_random):
        streaming = Random(args.sample_random_width, args.sample_random_height)
        
    else:
        
        if (args.sample_path):
            p = Path(args.sample_path)
            if (p.is_dir()):
                streaming = Sample(args.streaming_size).initdir(args.sample_path)
                
            elif (p.is_file()):
                streaming = Sample(args.streaming_size).initfile(args.sample_path)
    
            
    if (streaming is None):
        logger.error("no sample init. see --help")
        sys.exit()
    
    return streaming

def showasync(
    args: Arguments,
    windowname: str,
    runtime: RuntimeAsync,
    streaming: Streaming,
):
    
    f_feed = threading.Event()

    def __task_feed__():
        
        logger.info(f"streaming feed start {f_feed}")
        frame: cv2.typing.MatLike = None # type: ignore
        while True:
            f_feed.wait()
            time.sleep(0.001)

            feederr = 0
            while(f_feed.is_set() and feederr < 1000):
                frame = streaming.get()
                if frame is None:
                    feederr += 1
                    time.sleep(0.001)
                    continue
                
                else:
                    feederr = 0
                    break
                
            if(feederr >= 1000):
                # logger.error(f"frame capture on error")
                break
                
            while(f_feed.is_set() and runtime.avaliable < 1):
                time.sleep(0.001)
                
            runtime.put(
                InferenceSource(
                    frame,
                    args.confidence,
                    args.threshold,
                    time.time(),
                )
            )
                
        logger.info(f"streaming feed stop {f_feed}")
        
    
    f_feed.set()
    t_feed = threading.Thread(target=__task_feed__, daemon=True)
    t_feed.start()
    
    
    f_run = threading.Event()
    f_run.set()
    
    t_run = threading.Thread(target=runtime.run, daemon=True)
    t_run.start()
    
    err = 0
    last = 0
    
    
    logger.info(f"wait for first inference result ...")
    while(f_run.is_set() and runtime.avaliable >= runtime.size):
        time.sleep(0.001)
    logger.info(f"wait for first inference result done")
    
    while(f_run.is_set() and err < 1000):
        
        if(runtime.avaliable < 3):
            f_feed.clear()
        else:
            f_feed.set()
        
        result: InferenceResult = runtime.get()
        if(result is None):
            time.sleep(0.001)
            err += 1
            continue
        
        
        runtime.add_count()
        runtime.add_spendtime(result.spendtime)
        
        if (args.display and result.timestamp > last):
            last = result.timestamp
            image = result.image
            
            utils.drawmodelname(image=image, name=runtime.information.name)
            utils.drawlatency(image=image, spendtime=runtime.latency)
            utils.drawfps(image=image, framecount=runtime.fps)
            utils.drawsize(image=image, width=image.shape[1], height=image.shape[0])
            
            window_show(windowname, result.image)
            key = window_waitkey(1)
            if key == KEY_Q or key == KEY_q:
                break
        
        err = 0
                    
        if (not t_feed.is_alive()):
            break
        
    if (err > 0):
        logger.error(f"terminate show async err:{err}")
    
    runtime.clear()
    runtime.stop()
    
    f_run.clear()
    t_run.join(1)
    
    f_feed.clear()
    t_feed.join(1)
    
    logger.info(f"runtime stop: {t_run.is_alive()}")
    # logger.info(f"feed stop: {t_feed.is_alive()}")

def show(
    args: Arguments,
    windowname: str,
    runtime: Runtime,
    streaming: Streaming,
):
    
    err = 0
    while(err < 10000):
        
        frame = streaming.get()
        if(frame is None):
            time.sleep(0.001)
            err += 1
            continue
        
        err = 0
        
        try:
            
            result = runtime.inference(
                InferenceSource(
                    image=frame,
                    confidence=args.confidence,
                    threshold=args.threshold,
                    timestamp=time.time()
                )
            )
            
            runtime.add_count()
            runtime.add_spendtime(result.spendtime)
            
            if (args.display):
                image  = result.image
                
                utils.drawmodelname(image=image, name=runtime.information.name)
                utils.drawlatency(image=image, spendtime=runtime.latency)
                utils.drawfps(image=image, framecount=runtime.fps)
                utils.drawsize(image=image, width=image.shape[1], height=image.shape[0])
                
                window_show(windowname, result.image)
                key = window_waitkey(1)
                if key == KEY_Q or key == KEY_q:
                    break
            
        except Exception as e:
            logger.error(e)
            
def main(args: Arguments):
    
    # if (args.debug_leaks):
    #     tk = tracker.SummaryTracker()
    #     hp = guppy.hpy()
    #     hp.setrelheap()
    
    logger.info(args.__dict__)
    
    runtime = initruntime(args)
    
    streaming = initstreaming(args)
    
    if (args.monitor):
        monitor = Monitor(
            Path(args.model_path).name
            if args.model_path
            else "dryrun"
        )
        runtime.monitor = monitor
    
    windowname = f"{runtime.information.name} {streaming.name}"
    
    if (args.display):
        window_init(windowname)
    
    
    logger.info(f"startup {windowname}")
    
    while(True):
        
        streaming.start()
        monitor.start()
        
        if (isinstance(runtime, RuntimeAsync)):
            logger.info("startup runtime async")
            showasync(
                args,
                windowname,
                runtime,
                streaming,
            )
            
        elif (isinstance(runtime, Runtime)):
            logger.info("startup runtime")
            show(
                args,
                windowname,
                runtime,
                streaming,
            )
            

        streaming.stop()
        monitor.stop()
        
        if (args.debug_leaks):
            try:
                unreachable = gc.collect()
                logger.debug(f"Unreachable objects: {unreachable}")
                logger.debug(f"Garbage objects: {len(gc.garbage)}")
                for i, obj in enumerate(gc.garbage):
                    logger.debug(f"    [{i}] type={type(obj)} repr={repr(obj)[:100]}")
                
                monitor.dump_objects()
                
                # tk.print_diff()
                # items = hp.heap()[:10]
                # for i in range(len(items)):
                #     logger.debug(f"===== item[{i}]: kind: {items[i].kind} =====")
                #     logger.debug(f"\nbyvia: {items[i].byvia}")
                #     logger.debug(f"\nshpaths: {items[i].shpaths}")
                #     logger.debug(f"\nrp: {items[i].rp}")
                    
            except:
                logger.warning("debug memory leaks failed")
        
        if (args.loop):
            
            if (args.display):
                logger.info("wait for keypress in 3 seconds, arrow left = prev, right = next, default = next")
                key = window_waitkey(3000)
                if key == KEY_Q or key == KEY_q:
                    break
                
                if key == KEY_ARROW_LEFT:
                    streaming.prev()
                    
                if key == KEY_ARROW_RIGHT:
                    streaming.next()
                    
                if key < 0:
                    streaming.next()
            else:    
                streaming.next()
            
        else:
            break
        
    logger.info(f"shutdown {windowname}")
    window_close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", type=str, default="", help="sample filepath, can be directory or signle file")
    parser.add_argument("--sample-mjpeg", type=str, default="", help="cctv mjpeg url see: https://tisvcloud.freeway.gov.tw/history/motc20/CCTV.xml")
    parser.add_argument("--model-path", type=str, default="", help="model path")
    parser.add_argument("--confidence", type=int, default=50, help="confidence threshold")
    parser.add_argument("--threshold", type=int, default=50, help="nms filter threshold")
    parser.add_argument("--streaming-size", type=int, default=64, help="the limit of video streaming buffer size")
    parser.add_argument("--display", action="store_true", help="display inference results")
    parser.add_argument("--monitor", action="store_true", help="monitor inference frame per second when input sample is video")
    parser.add_argument("--loop", action="store_true", help="loop forever when input sample is video")
    parser.add_argument("--sample-random", action="store_true", help="auto generate random buffer array as inference source")
    parser.add_argument("--sample-random-width", type=int, default=320, help="with --sample-random buffer width size")
    parser.add_argument("--sample-random-height", type=int, default=320, help="with --sample-random buffer height size")

    parser.add_argument("--no-inference", action="store_true", help="no inference consumer")
    parser.add_argument("--debug-leaks", action="store_true")

    args = parser.parse_args()
    main(Arguments(**vars(args)))