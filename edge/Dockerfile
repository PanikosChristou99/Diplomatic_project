FROM base_image:latest

COPY . /app

WORKDIR /app

RUN pip3 install flask[async]

# RUN pip3 install psutil

ARG Port

# EXPOSE ${Port}

CMD ["python3", "app.py"]
