import argparse
import os

import tensorflow as tf
from face_emotion.io.json import dump_json
from face_emotion.data.dataset import build_tf_dataset
from face_emotion.models.models import build_from_name, supported_models

from common import N_CLASSES, INPUT_SHAPE, get_df

def get_args():
    parser = argparse.ArgumentParser(
        prog='FaceEmotionTrainer',
        description='Train a model for face emotion classification.')

    parser.add_argument(
        '-t', '--train_path',
        type=str, required=True,
        help='Path to the training dataset.')
    parser.add_argument(
        '-v', '--val_path',
        type=str, required=True,
        help='Path to the validation dataset.')
    parser.add_argument(
        '-o', '--output_path',
        type=str, required=False,
        default='outputs/',
        help='Output path for saving models and logs.')
    parser.add_argument(
        '-b', '--batch_size',
        type=int, required=False, default=512,
        help='Training batch size.')
    parser.add_argument(
        '-e', '--epochs',
        type=int, required=False, default=50,
        help='Total number of epochs.')
    parser.add_argument(
        '-lr', '--lerning_rate',
        type=float, required=False, default=0.001,
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

def get_loss(loss_f: str):
    # categorical focal ce cannot be initialized as a string
    if loss_f == 'categorical_focal_crossentropy':
        return tf.keras.losses.CategoricalFocalCrossentropy()

    return loss_f

def get_callbacks(out_path: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_path, 'face_emotion_clf.weights.h5'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_f1_score',
            mode = 'max'
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(out_path, 'logs.csv'))
    ]

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    dump_json(os.path.join(args.output_path, 'args.json'), vars(args))
    
    # gets the model class and custom normalization function
    model, normalization_f = build_from_name(
        args.model, n_classes=N_CLASSES, input_shape=INPUT_SHAPE
    )

    # loads the label dataframes from CSV files
    print('Reading data...')
    train_df = get_df(args.train_path)
    val_df = get_df(args.val_path)
    
    print('Creating dataset...')
    # creates the datasets
    comm_param = {
        'n_classes': N_CLASSES,
        'batch_size': args.batch_size,
        'norm_f': normalization_f,
        'img_shape': INPUT_SHAPE
    }
    train_ds = build_tf_dataset(train_df, shuffle=True, **comm_param)
    val_ds = build_tf_dataset(val_df, shuffle=False, **comm_param)
    
    print('Compiling model...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lerning_rate),
        loss=get_loss(args.loss_f),
        metrics=[
            'accuracy',
            tf.keras.metrics.F1Score(average='weighted')
        ]
    )
    print('Running training...')
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=get_callbacks(args.output_path)
    )