import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from face_emotion.data.dataset import build_tf_dataset
from face_emotion.io.json import read_json, dump_json
from face_emotion.models.models import build_from_name

from common import N_CLASSES, INPUT_SHAPE, LABELS, get_df

def plot_cm(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eval_path: str = 'eval/',
        label_ids: list = list(range(N_CLASSES)),
        label_names: list = LABELS):
    cm_config = {'labels': label_ids, 'display_labels': label_names}
    cmp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize='true', **cm_config
    )
    _, ax = plt.subplots(figsize=(10,10))
    cmp.plot(ax=ax)
    plt.savefig(os.path.join(eval_path, 'norm_conf_mtx.png'))

    plt.clf()
    cmp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize=None, **cm_config
    )
    _, ax = plt.subplots(figsize=(10,10))
    cmp.plot(ax=ax)
    plt.savefig(os.path.join(eval_path, 'conf_mtx.png'))

def generate_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_count: np.ndarray,
        eval_path: str = 'eval/',
        label_ids: list = list(range(N_CLASSES)),
        label_names: list = LABELS):
    report = classification_report(
        y_true, y_pred,
        digits=4, output_dict=True,
        labels=label_ids, target_names=label_names
    )
    report = {
        'fps': y_true.shape[0] / time_count.sum(),
        'latency': time_count.sum() / y_true.shape[0],
        'latency_p_batch': float(time_count.mean()),
        **report
    }
    dump_json(os.path.join(eval_path, 'report.json'), report)

def evaluate_model(model, test_ds):
    full_y_pred = []
    full_y_true = []
    time_count = []
    for X_data, y_true in tqdm(test_ds):
        # run prediction
        _start = time.time()
        y_pred = model(X_data, training=False)
        _end = time.time() - _start

        # compute execution time and save predictions
        time_count.append(_end)
        full_y_true.extend(np.argmax(y_true, axis=1).tolist())
        full_y_pred.extend(np.argmax(y_pred, axis=1).tolist())

    return (
        np.array(full_y_true, dtype=np.int32),
        np.array(full_y_pred, dtype=np.int32),
        np.array(time_count, dtype=np.float32)
    )

def get_args():
    parser = argparse.ArgumentParser(
        prog='FaceEmotionEval',
        description='Evaluate a trained Face Emotion recognition model.')

    parser.add_argument(
        '-t', '--test_path',
        type=str, required=True,
        help='Path to the test dataset.')
    parser.add_argument(
        '-e', '--exp_path',
        type=str, required=True,
        help='Path to the experiment folder (train output).')
    parser.add_argument(
        '-b', '--batch_size',
        type=int, required=False, default=32,
        help='Testing batch size.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train_config = read_json(os.path.join(args.exp_path, 'args.json'))

    model, normalization_f = build_from_name(train_config['model'])
    model.build(input_shape=(None, *INPUT_SHAPE))
    model.load_weights(
        os.path.join(args.exp_path, 'face_emotion_clf.weights.h5')
    )
    
    test_df = get_df(args.test_path)
    test_ds = build_tf_dataset(
        test_df,
        shuffle=False,
        n_classes=N_CLASSES,
        batch_size=args.batch_size,
        norm_f=normalization_f,
        img_shape=INPUT_SHAPE)
    y_true, y_pred, time_count = evaluate_model(model, test_ds)

    eval_path = os.path.join(args.exp_path, 'eval')
    os.makedirs(eval_path, exist_ok=True)
    label_ids = np.unique(y_true)
    label_names=[LABELS[i] for i in label_ids]
    # generate reports
    generate_report(
        y_true, y_pred, time_count,
        eval_path=eval_path, label_ids=label_ids, label_names=label_names
    )
    # save confusion matrix
    plot_cm(
        y_true, y_pred,
        eval_path=eval_path, label_ids=label_ids, label_names=label_names
    )