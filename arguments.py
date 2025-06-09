

from dataclasses import dataclass


@dataclass
class Arguments:
    
    sample_path: str = ""
    sample_mjpeg: str = ""
    
    model_path: str = ""
    
    confidence: int = 10
    threshold: int = 10
    
    display: bool = False
    loop: bool = False
    
    streaming_size: int = 8
    
    no_inference: bool = False
    
    sample_random: bool = False
    sample_random_width: int = 320
    sample_random_height: int = 320