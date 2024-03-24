# Facial Emotion Recognition

This project aims to create a face emotion recognition deep learing model. In order to achieve such goal, the [FER+](https://github.com/microsoft/FERPlus) dataset was employed for training, validation and testing.

Details on how to setup the project and run the main scripts, are described as follows.

## Project Organization
This project is organized as follows:
- `data`: path containing all of the data used for training, evaluation and testing;
- `face_emotion`: library containing the core structures, as: models, dataset creations, auxiliary functions, etc.;
- `notebooks`: Jupyter notebooks developed for data exploration and problem understanding;
- `scripts`: training and evaluation scripts.

## Env Setup and Library Installation
In order to build the project and run the training/evaluation scripts, we recommend that a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment is created with Python v3.10 and then activated:

```shell
conda create -n face_emotion python==3.10
conda activate face_emotion
```

In this project, we developed a small library called `face_emotion`, in which has its dependencies and building process menaged by [Poetry](https://python-poetry.org). Such package can be easily installed with:

```shell
pip install poetry
```

The library dependencies are described in `pyproject.toml`. In this one, it is possible to observe that the library requires the following packages:

```
opencv-python==4.9.0.80
tensorflow==2.16.1
albumentations==1.4.2
matplotlib==3.8.3
pandas==2.2.1
tqdm==4.66.2
```

In order to install them and the `face_emotion` library, it is required that the following commands are executed:

```
poetry build
pip install dist/face_emotion-0.1.0-py3-none-any.whl
```
