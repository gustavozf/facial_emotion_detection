# Facial Emotion Recognition

This project aims to create a face emotion recognition deep learing model. In order to achieve such goal, the [FER+](https://github.com/microsoft/FERPlus) dataset was employed for training, validation and testing.

Details on how to setup the project and run the main scripts, are described as follows.

## Project Organization
This project is organized as follows:
- `data`: path containing all of the data used for training, evaluation and testing (to donwload the full dataset, please recall the [FER+](https://github.com/microsoft/FERPlus) repository);
- `deploy`: sample application developed in flask to showcase the model usage in a production level;
- `docs`: files containing information about the taken experiments;
- `face_emotion`: library containing the core structures, as: models, dataset creations, auxiliary functions, etc.;
- `notebooks`: Jupyter notebooks developed for data exploration and problem understanding;
- `scripts`: training and evaluation scripts.

## Env Setup and Library Installation
In order to build the project and run the training/evaluation scripts, we recommend that a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment is created with Python v3.10:

```shell
conda create -n face_emotion python==3.10
conda activate face_emotion
```

In this project, we developed a small library called `face_emotion`, in which has its dependencies and building process menaged by [Poetry](https://python-poetry.org). Such package can be easily installed with:

```shell
pip install poetry
```

The library dependencies are described in `pyproject.toml`. In this project, our library requires the following packages:

```
opencv-python==4.9.0.80
tensorflow==2.16.1
albumentations==1.4.2
matplotlib==3.8.3
pandas==2.2.1
tqdm==4.66.2
```

In order to install them and the `face_emotion` library, plese execute the following commands:

```
poetry build
pip install dist/face_emotion-0.1.2-py3-none-any.whl
```
## Training and Evaluation
Both training and evaluation scripts may be found under the `scripts/` folder. A sample command for using both of the training and evaluation scripts, may be seen as follows:

```
DATA_PATH="/path/to/data"
OUT_PATH="/path/to/outputs"

python train.py \
    --train_path "$DATA_PATH/FER2013Train" \
    --val_path "$DATA_PATH/FER2013Valid" \
    --output_path "$OUT_PATH" \
    --batch_size 1024 \
    --epochs 50 \
    --lerning_rate 0.001 \
    --model effnetb1 \
    --loss_f categorical_crossentropy \
    --data_aug strong

python eval.py \
    --test_path "$DATA_PATH/FER2013Test" \
    --exp_path "$OUT_PATH" \
    --batch_size 512
```

## Deployment
A sample flask application was developed and made available in `deploy/`. In order to run it, it is required that the following packages are installed:

```
flask==3.0.2
jsonpickle==3.0.3
```

The following command need to be executed to start the application:
```
flask run
```

Although a Dockerfile is made available for containerization, it is worth mentioning that it is yet to be validated.