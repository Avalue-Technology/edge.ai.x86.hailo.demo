
import logging
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy

from frame.streaming import Streaming

logger = logging.getLogger(__name__)

class Random(Streaming):
    
    def __init__(self, width: int, height: int, length: int = 100) -> None:
        super().__init__(length)
        
        self._width = width
        self._height = height
        self._length = length

        self._frames: List[cv2.typing.MatLike] = []
        self.__init_random_frames__()
        
        self._streaming_running = threading.Event()
        self._streaming = None
        
        self._name = f"Random frames {width}x{height}"
        
    def __create_frame__(self, width: int, height: int):
        return (numpy.random.rand(height, width, 3) * 255).astype(numpy.uint8)
    
    def __init_random_frames__(self) -> None:
        
        size = self._length
        __pool_random__ = ThreadPoolExecutor(
            thread_name_prefix="__pool_random__",
            max_workers=self._length
            if self._length < 100
            else 100
        )
        
        logger.info("init frames")
        while(True):
            if (size <= 0):
                break
            
            future = __pool_random__.submit(
                self.__create_frame__,
                self._width,
                self._height
            )
            self._frames.append(future.result())
            size -= 1
        
        logger.info("wait for all task done ...")
        __pool_random__.shutdown()
        logger.info("wait for all task done ok")
            
    def __task_streaming__(self) -> None:
        index = 0
        
        logger.info("start streaming random frames")
        
        while(self._streaming_running.is_set()):
            frame = self._frames[index]

            err = 0
            # logger.debug(f"wait for streaming buffer avaliable ...{self.avaliable}")
            while(not self.avaliable):
                time.sleep(0.001)
                err += 1
                continue
            # logger.debug(f"wait for streaming buffer avaliable ok {self.avaliable}")
            
            self.put(frame)
            self.__add_framecount__()
            
            # if (err > 0):
            #     logger.warning(f"add latency: {err}ms")
            
            err = 0
            
            index = (index + 1) % self._length
            
        self.clear()
        
        logger.info("stop streaming random frames")
            
    def start(self) -> None:
        self._streaming_running.set()
        logger.info(f"startup random: {self._width}x{self._height} {self._streaming_running}")
        
        if (self._streaming is None):
            self._streaming = threading.Thread(
                target=self.__task_streaming__,
                daemon=True,
            )
            
        self._streaming.start()
        return super().start()
        
    def stop(self) -> None:
        self._streaming_running.clear()
        logger.info(f"shutdown random: {self._width}x{self._height} {self._streaming_running}")
        
        if (
            self._streaming is not None
            and self._streaming.is_alive()
        ):
            self._streaming.join(1)
        
        self._streaming = None
        return super().stop()