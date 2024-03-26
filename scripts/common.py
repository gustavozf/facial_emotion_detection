import os

import pandas as pd
import numpy as np

LABELS = [
    'neutral',
    'happiness',
    'surprise',
    'sadness',
    'anger',
    'disgust',
    'fear',
    'contempt',
    'unknown',
    'not_a_face'
]
DF_COLUMNS = ['img_name', 'bbox', *LABELS]
N_CLASSES = len(LABELS)
INPUT_SHAPE = (48, 48, 3)
LABELS_FNAME = 'label.csv'

def get_df(data_path: str) -> pd.DataFrame:
    ''' Read the labels CSV as a DataFrame. '''
    df = pd.read_csv(
        os.path.join(data_path, LABELS_FNAME),
        header=None, index_col=None, names=DF_COLUMNS)
    # expand the image names to the full path
    df['img_name'] = data_path + os.sep + df['img_name']
    # generate the label based on the annotations
    df['label'] = np.argmax(df.iloc[:, 2:], axis=1)
    # remove redundant columns
    # since all bboxes have the same value, they all can be removed
    df = df.drop(LABELS + ['bbox'], axis=1)

    return df