#!/bin/bash

source venv/bin/activate

python3 ./main.py \
--monitor \
--display \
--loop \
--sample-path=sdk/samples/videos \
--model-path=sdk/models/object-detection/yolo/hailo-8-hef/yolov11x.hef \
