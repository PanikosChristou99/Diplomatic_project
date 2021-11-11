FROM workloader:latest

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt 

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]
