#!/bin/bash

DATA_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/data"
OUT_PATH="/Volumes/Extreme SSD/facial_emotion_recognition/outputs/v008"

python train.py \
    --train_path "$DATA_PATH/FER2013Train" \
    --val_path "$DATA_PATH/FER2013Valid" \
    --output_path "$OUT_PATH" \
    --batch_size 1024 \
    --epochs 50 \
    --lerning_rate 0.001 \
    --model effnetb1 \
    --loss_f categorical_crossentropy \
    --data_aug strong

python eval.py \
    --test_path "$DATA_PATH/FER2013Test" \
    --exp_path "$OUT_PATH" \
    --batch_size 512
