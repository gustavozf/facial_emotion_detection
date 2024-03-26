#!/bin/bash

DATA_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/data"
OUT_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/outputs"

python eval.py \
    --test_path "$DATA_PATH/FER2013Test" \
    --exp_path "$OUT_PATH/v001_v2" \
    --batch_size 512