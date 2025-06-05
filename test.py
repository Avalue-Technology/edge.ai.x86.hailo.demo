
import logging
import time

from frame.random import Random
from sdk.data.circular_buffer import CircularBuffer


from pympler import muppy, summary, tracker, classtracker
import gc

mem_tracker = tracker.SummaryTracker()
buffer_tracker = classtracker.ClassTracker()
buffer_tracker.track_class(CircularBuffer)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)10s %(levelname)5s] %(filename)s.%(funcName)s:%(lineno)d: %(message)s"
)

logger = logging.getLogger(__name__)

streaming = Random(1920, 1080, 100)
buffer = CircularBuffer(100)
buffer_tracker.create_snapshot()

mem_tracker.print_diff()

streaming.start()

l = 10000
while l > 0:
    
    f = streaming.get()
    if f is None:
        time.sleep(0.001)
        continue
        
    if buffer.avaliable > 1:
        buffer.put(f)
        
    else:
        buffer.get()
        
    # if (l % 100 == 0):
    #     logger.debug(f"... {l}")
        
    buffer_tracker.stats.print_summary()
    l -= 1

buffer.clear()
mem_tracker.print_diff()
streaming.stop()