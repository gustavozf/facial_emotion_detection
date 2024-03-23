import tensorflow as tf

@tf.function
def pass_through_norm(img):
    return img

def keras_initializer(model_class, input_shape: tuple = (48, 48, 3)):
    return model_class(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

MODELS = {
    'mobilenet': tf.keras.applications.MobileNetV2, # 3.5M params
    'effnetb0': tf.keras.applications.EfficientNetV2B0, # 7.2M params
    'effnetb1': tf.keras.applications.EfficientNetV2B1, # 8.2M params
    'effnetb2': tf.keras.applications.EfficientNetV2B2, # 10.2M params
    'convnexttiny': tf.keras.applications.ConvNeXtTiny, # 28.6M params
    'resnet50': tf.keras.applications.ResNet50V2 # 25.6M params
}
NORMALIZATIONS = {
    'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input,
    'effnetb0': pass_through_norm,
    'effnetb1': pass_through_norm,
    'effnetb2': pass_through_norm,
    'convnexttiny': pass_through_norm,
    'resnet50': tf.keras.applications.resnet_v2.preprocess_input
}
INITIALIZERS = {
    'mobilenet': keras_initializer,
    'effnetb0': keras_initializer,
    'effnetb1': keras_initializer,
    'effnetb2': keras_initializer,
    'convnexttiny': keras_initializer,
    'resnet50': keras_initializer
}

supported_models = set(MODELS.keys())

def select_model_class_by_name(model_name: str):
    if model_name not in supported_models:
        raise ValueError(
            f'Model "{model_name}" not supported.'
            f' Please use one of: {supported_models}'
        )
    
    return MODELS[model_name], NORMALIZATIONS[model_name]

def build_from_name(
        model_name: str,
        n_classes: int = 10,
        input_shape: tuple = (48, 48, 3),
    ):
    model_class, norm_f = select_model_class_by_name(model_name)

    model = tf.keras.Sequential([
        INITIALIZERS[model_name](model_class, input_shape=input_shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    return model, norm_f