from typing import Callable
from functools import partial

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from face_emotion.data.data_aug import DataAugmenter
from face_emotion.models.models import pass_through_norm as default_norm

@tf.numpy_function(Tout=(tf.uint8, tf.int64))
def read_gs_image(img_path, label):
    img = cv2.imread(img_path.decode("utf-8"), 0)

    # creates a 3-channel grayscale image
    gs_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    gs_img[:, :, 0] = img
    gs_img[:, :, 1] = img
    gs_img[:, :, 2] = img
    del img

    return gs_img, label

@tf.function
def set_shape(img, label, img_shape: tuple = (48, 48, 3)):
    img.set_shape(img_shape)
    label.set_shape([])
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int64)
    return img, label

@tf.function
def normalize_img(img_batch, label_batch, norm_f: Callable = default_norm):
    return norm_f(img_batch), label_batch

@tf.function
def label_to_one_hot(img_batch, label_batch, n_classes: int = 10):
    return img_batch, tf.one_hot(label_batch, n_classes)

def build_tf_dataset(
        labels_df: pd.DataFrame,
        batch_size: int = 32,
        shuffle: bool = False,
        data_aug: str = None,
        n_classes: int = 10,
        img_shape: tuple = (48, 48, 3),
        norm_f: Callable = default_norm
    ) -> tf.data.Dataset:
    ''' Get the input CSV and create a tf.data.DataSet '''
    
    ds = tf.data.Dataset.from_tensor_slices(
        (labels_df['img_name'], labels_df['label'])
    )
    
    if shuffle:
        # shuffle with a buffer == len(dataset)
        ds = ds.shuffle(labels_df.shape[0])

    ds = ds.map(read_gs_image, num_parallel_calls=tf.data.AUTOTUNE)

    if data_aug is not None:
        augmenter = DataAugmenter(aug_type=data_aug, img_size=img_shape[0])
        ds = ds.map(
            lambda img, label: (
                tf.numpy_function(func=augmenter, inp=[img], Tout=tf.uint8),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        partial(set_shape, img_shape=img_shape),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(
        partial(normalize_img, norm_f=norm_f),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        partial(label_to_one_hot, n_classes=n_classes),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds