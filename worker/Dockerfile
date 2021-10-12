FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY . /app

# RUN mkdir /app

# COPY ./main.py /app

# COPY ./requirements.txt /app

WORKDIR /app

RUN pip3 install -r requirements.txt

# RUN pip3 install kaggle

# RUN mkdir /root/.kaggle

# RUN cp ./kaggle.json /root/.kaggle/

# RUN kaggle datasets download -d pavansanagapati/images-dataset --force

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]
