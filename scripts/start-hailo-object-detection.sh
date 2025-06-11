#!/bin/bash

source venv/bin/activate

HAILO_MODULE=$1
MODEL_NAME=$2

SAMPLE_PATH=sdk/samples/videos
MODEL_PATH=sdk/models/object-detection/yolo/$HAILO_MODULE-hef/$MODEL_NAME.hef

if [[ -z "$HAILO_MODULE" || -z "$MODEL_NAME" ]]; then
    echo "unspecified model path or module name. please use:" 
    echo "  example 1: running hailo-8 with yolov8n:"
    echo "    start-hailo-object-detection.sh <hailo-8> <yolov8n>"
    echo ""
    echo "  example 2: running hailo-8l with yolov11n:"
    echo "    start-hailo-object-detection.sh <hailo-8l> <yolov11n>"

    exit 1
fi

echo "start running $HAILO_MODULE object detection for model $MODEL_PATH"

while true
do
    for file in "$SAMPLE_PATH"/*.mp4;
    do
        python3 ./main.py \
            --display \
            --full-screen \
            --streaming-size=8 \
            --sample-path=$file \
            --model-path=$MODEL_PATH
    done
done


