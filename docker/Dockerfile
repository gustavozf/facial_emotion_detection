FROM tensorflow/tensorflow:2.16.1

COPY resources/ resources/
COPY app.py app.py

RUN apt update && install llibgl1 ibglib2.0-0
RUN pip install --ignore-installed flask==3.0.2 jsonpickle==3.0.3
RUN pip install resources/face_emotion-0.1.2-py3-none-any.whl

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
