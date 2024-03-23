import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from face_emotion.data.dataset import build_tf_dataset
from face_emotion.models.models import (
    select_model_class_by_name, supported_models
)

from consts import DF_COLUMNS, N_CLASSES, LABELS, INPUT_SHAPE, LABELS_FNAME

def get_args():
    parser = argparse.ArgumentParser(
        prog='FaceEmotionTrainer',
        description='Train a model for face emotion classification.',
        epilog='Text at the bottom of help')

    parser.add_argument(
        't', '--train_path',
        type=str, required=True,
        help='Path to the training dataset.')
    parser.add_argument(
        'v', '--val_path',
        type=str, required=True,
        help='Path to the validation dataset.')
    parser.add_argument(
        '-o', '--output_path',
        type=str, required=False,
        default='outputs/',
        help='Output path for saving models and logs.')
    parser.add_argument(
        '-b', '--batch_size',
        type=int, required=False, default=32,
        help='Training batch size.')
    parser.add_argument(
        '-e', '--epochs',
        type=int, required=False, default=100,
        help='Total number of epochs.')
    parser.add_argument(
        '-lr', '--lerning_rate',
        type=float, required=False, default=0.01,
        help='Initial learning rate.')
    parser.add_argument(
        '-m', '--model',
        type=str, required=False,
        default='mobilenet', choices=supported_models,
        help='Initial learning rate.')
    parser.add_argument(
        '-l', '--loss_f',
        type=str, required=False,
        default='categorical_crossentropy',
        help='Loss function.')
    
    return parser.parse_args()

def get_df(data_path: str) -> pd.DataFrame:
    ''' Read the labels CSV as a DataFrame. '''
    df = pd.read_csv(
        os.path.join(data_path, LABELS_FNAME),
        header=None, index_col=None, names=DF_COLUMNS)
    # expand the image names to the full path
    df['img_name'] = data_path + os.sep + df['img_name']
    # generate the label based on the annotations
    df['label'] = np.argmax(df.iloc[:, -N_CLASSES:], axis=1)
    # remove redundant columns
    # since all bboxes have the same value, they all can be removed
    df = df.drop(LABELS + ['bbox'], axis=1)

    return df

def get_callbacks(out_path: str):
    os.makedirs(out_path, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_path, 'face_emotion_clf.keras'),
            save_best_only=True,
            monitor='val_f1_score',
            mode = 'max'
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_path, 'logs.csv'))
    ]

if __name__ == '__main__':
    args = get_args()
    
    # gets the model class and custom normalization function
    model_class, normalization_f = select_model_class_by_name(args.model)

    # loads the label dataframes from CSV files
    train_df = get_df(args.train_path)
    val_df = get_df(args.val_path)
    
    # creates the datasets
    comm_param = {
        'n_classes': N_CLASSES,
        'batch_size': args.batch_size,
        'norm_f': normalization_f
    }
    train_ds = build_tf_dataset(train_df, shuffle=True, **comm_param)
    val_ds = build_tf_dataset(val_df, shuffle=False, **comm_param)
    
    # loads the base model and compiles it
    model = model_class(
        include_top=True,
        weights='imagenet',
        input_shape=INPUT_SHAPE,
        classes=N_CLASSES,
        classifier_activation='softmax'
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lerning_rate),
        loss=args.loss_f, metrics=['accuracy', 'f1_score']
    )
    # run the training session
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=get_callbacks(args.output_path)
    )