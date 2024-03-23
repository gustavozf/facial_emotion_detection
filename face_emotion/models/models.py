import tensorflow as tf

@tf.function
def pass_through_norm(img):
    return img

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

supported_models = set(MODELS.keys())

def select_model_class_by_name(model_name: str):
    if model_name not in supported_models:
        raise ValueError(
            f'Model "{model_name}" not supported.'
            f' Please use one of: {supported_models}'
        )
    
    return MODELS[model_name], NORMALIZATIONS[model_name]