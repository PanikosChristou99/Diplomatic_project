FROM base_image:latest

COPY . /app

WORKDIR /app

RUN pip3 install hwcounter 

ARG Port

# EXPOSE ${Port}

CMD ["python3", "app.py"]
