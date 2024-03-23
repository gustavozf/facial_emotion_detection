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
INPUT_SHAPE = (48, 48)
LABELS_FNAME = 'label.csv'