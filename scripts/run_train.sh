#!/bin/bash

DATA_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/data"
OUT_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/outputs"

python train.py \
    --train_path "$DATA_PATH/FER2013Train" \
    --val_path "$DATA_PATH/FER2013Valid" \
    --output_path "$OUT_PATH/v001_v2" \
    --batch_size 512 \
    --epochs 50 \
    --lerning_rate 0.001 \
    --model effnetb2 \
    --loss_f categorical_crossentropy
