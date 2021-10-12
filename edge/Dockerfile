FROM tensorflow/tensorflow:latest

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY . /app

WORKDIR /app

RUN pip3 install tensorflow_hub

RUN pip3 install keras

RUN pip3 install bottle

RUN pip3 install Pillow

# RUN pip3 install tensorflow

RUN pip3 install matplotlib

RUN pip3 install numpy

EXPOSE 5000

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
