FROM jjanzic/docker-python3-opencv:latest

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev zlib1g-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY . /app

# RUN mkdir /app

# COPY ./main.py /app

# COPY ./requirements.txt /app

WORKDIR /app

RUN apt-get install libjpeg-dev -y

RUN pip3 install --upgrade setuptools wheel

RUN pip3 install scikit-build

RUN pip3 install cmake

RUN pip3 install --upgrade Pillow

RUN pip3 install -r requirements.txt 

RUN python3 download_dataset.py

# RUN mkdir /root/.kaggle

# RUN cp ./kaggle.json /root/.kaggle/

# RUN kaggle datasets download -d pavansanagapati/images-dataset --force

# ENTRYPOINT [ "python3" ]

# CMD [ "main.py" ]
