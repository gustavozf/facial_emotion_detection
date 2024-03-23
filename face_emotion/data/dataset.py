from typing import Callable
from functools import partial

import cv2
import pandas as pd
import tensorflow as tf

from face_emotion.models.models import pass_through_norm as default_norm

@tf.numpy_function(Tout=(tf.uint8, tf.int64))
def read_gs_image(img_path, label):
    return cv2.imread(img_path.decode("utf-8"), 0), label

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
        n_classes: int = 10,
        norm_f: Callable = default_norm
    ) -> tf.data.Dataset:
    ''' Get the input CSV and create a tf.data.DataSet '''
    
    ds = tf.data.Dataset.from_tensor_slices(labels_df)
    
    if shuffle:
        # shuffle with a buffer == len(dataset)
        ds = ds.shuffle(labels_df.shape[0])

    ds = ds.map(read_gs_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        partial(label_to_one_hot, n_classes=n_classes),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds