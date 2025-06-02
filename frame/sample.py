

from dataclasses import dataclass
from pathlib import Path

import time
import logging
import threading

from typing import Self

import cv2

from frame.streaming import Streaming
from sdk.commons import utils

logger = logging.getLogger(__name__)

class Sample(Streaming):
    
    def __init__(self, size: int):
        super().__init__(size)
        
        self._filepath = ""
        self._filelength = 0
        self._fileindex = 0
        self._files = []
        
        self._streaming_running = threading.Event()
        self._streaming = None # Thread(target=self.__task_capture__)
        
    def __task_streaming__(self, file: str):
        
        capture = cv2.VideoCapture(file)
        if (not capture.isOpened()):
            logger.error("video capture is not opened")
            return None
        
        err = 0
        
        # logger.debug(f"start video capture {file} {self._streaming_running}")
        
        while(
            capture.isOpened()
            and self._streaming_running.is_set()
        ):
            ret, frame = capture.read()
            if (not ret or frame is None):
                err += 1
                if (err >= 10000):
                    logger.error(f"failed to read frame from video {file} {err}")
                    break
                
                time.sleep(0.001)
                continue
            
            err = 0

            # logger.debug(f"wait for streaming buffer avaliable ...{self.avaliable}")
            while(self.avaliable < 1):
                time.sleep(0.001)
                err += 1
                continue
            # logger.debug(f"wait for streaming buffer avaliable ok {self.avaliable}")
            
            self.put(frame)
            self.__add_framecount__()
            
            # if (err > 0):
            #     logger.warning(f"add latency: {err}ms")
                
            err = 0
    
        capture.release()
        self.clear()
        
        logger.info(f"release: {file} ok")
        
    def initdir(self, filepath: str) -> Self:
        self._filepath = filepath
        
        self._files = utils.fileslist(filepath)
        self._filelength = len(self._files)
        self._fileindex = 0
        self._file = self._files[self._fileindex]
        
        self._name = f"Sample file: {self._filepath}"
        
        return self
        
    def initfile(self, filepath: str) -> Self:
        self._filepath = filepath
        
        self._files = [self._filepath]
        self._filelength = 1
        self._fileindex = 0
        self._file = self._files[self._fileindex]
        
        self._name = f"Sample file: {self._filepath}"
        
        return self
        
    def next(self) -> None:
        self._fileindex = (self._fileindex + 1) % self._filelength
        self._file = self._files[self._fileindex]
        
        
    def prev(self) -> None:
        self._fileindex = (self._fileindex - 1 + self._filelength) % self._filelength
        self._file = self._files[self._fileindex]
    
    def start(self) -> None:
        self._streaming_running.set()
        logger.info(f"startup capture: {self._file} {self._streaming_running}")
        
        if (self._streaming is None):
            self._streaming = threading.Thread(
                target=self.__task_streaming__,
                args=(self._file,),
                daemon=True,
            )
            
        self._streaming.start()
        return super().start()
        
    def stop(self) -> None:
        self._streaming_running.clear()
        logger.info(f"shutdown capture: {self._file} {self._streaming_running}")
        
        if (
            self._streaming is not None
            and self._streaming.is_alive()
        ):
            self._streaming.join(1)
        
        self._streaming = None
        return super().stop()