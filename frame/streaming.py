

from abc import abstractmethod
import logging
import threading
import time
import cv2

from sdk.data.circular_buffer import CircularBuffer

logger = logging.getLogger(__name__)

class Streaming():
    
    def __init__(self, size: int) -> None:
        
        self._framecount = 0
        self._queue_frame = CircularBuffer(size)
        
        self._monitor_running = threading.Event()
        self._monitor = None # Thread(target=self.__task_monitor__)

        self._name = ""

    def __add_framecount__(self) -> None:
        self._framecount += 1
        
    def __task_monitor__(self):
        while(self._monitor_running.is_set()):
            # logger.info(f"frame count per second: {self._name} {self._framecount} ")
            self._framecount = 0
            time.sleep(1)
            
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def framecount(self) -> int:
        return self._framecount
    
    @property
    def avaliable(self) -> int:
        return self._queue_frame.avaliable

    def put(self, frame: cv2.typing.MatLike) -> int:
        return self._queue_frame.put(frame)
        
    def get(self) -> cv2.typing.MatLike:
        return self._queue_frame.get()
    
    def clear(self) -> None:
        self._queue_frame.clear()
    
    @abstractmethod
    def next(self) -> None:
        pass
    
    @abstractmethod
    def prev(self) -> None:
        pass
    
    @abstractmethod
    def start(self) -> None:
        self._monitor_running.set()
        self.clear()
        
        logger.info(f"startup monitor: {self._name} {self._monitor_running}")
        
        if (self._monitor is None):
            self._monitor = threading.Thread(
                target=self.__task_monitor__,
                daemon=True,
            )
        self._monitor.start()
    
    @abstractmethod
    def stop(self) -> None:
        self._monitor_running.clear()
        self.clear()
        
        logger.info(f"shutdown monitor: {self._name} {self._monitor_running}")
        
        if (
            self._monitor is not None
            and self._monitor.is_alive()
        ):
            self._monitor.join(1)
        self._monitor = None