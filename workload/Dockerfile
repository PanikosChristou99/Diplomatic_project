FROM base_image:latest

COPY . /app

WORKDIR /app

RUN pip3 install hwcounter 

CMD ["python3", "main.py"]
