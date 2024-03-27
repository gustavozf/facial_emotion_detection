import cv2
import jsonpickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response
from face_emotion.models.models import build_from_name

WEIGHTS_PATH = 'resources/face_emotion_clf.weights.h5'
MODEL_TYPE = 'effnetb1'
INPUT_SHAPE = (48, 48, 3)
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

# load model
print('Loading prediction model...')
model, normalization_f = build_from_name(MODEL_TYPE)
model.build(input_shape=(None, *INPUT_SHAPE))
model.load_weights(WEIGHTS_PATH)

# warmup model
print('Warming up model...')
model(np.zeros((1, *INPUT_SHAPE)), training=False)

# define app
app = Flask(__name__)

def pre_process(img):
    # creates a 3-channel grayscale image
    gs_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    gs_img[:, :, 0] = img
    gs_img[:, :, 1] = img
    gs_img[:, :, 2] = img
    img = gs_img

    img = tf.cast(img, tf.float32)

    return normalization_f(img)[tf.newaxis, ...]

@app.route("/predict")
def face_emotion_predict():
    try:
        img = np.fromstring(request.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = pre_process(img)
        preds = model(img, training=False)
    except Exception as e:
        return Response(
            response=jsonpickle.encode({
                'message': 'Error while processing image.',
                'error': str(e)
            }),
            status=500,
            mimetype='application/json'
        )

    pred = np.argmax(preds, axis=1).astype(int).tolist()[0]
    probs = preds.numpy().astype(float).tolist()[0]
    return Response(
        response=jsonpickle.encode({
            'message': 'Prediction successfully executed.',
            'prediction': pred,
            'pred_label': LABELS[pred],
            'probs': { LABELS[i]: prob for i, prob in enumerate(probs) }
        }),
        status=200,
        mimetype='application/json'
    )