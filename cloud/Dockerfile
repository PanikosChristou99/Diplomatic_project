FROM workloader:latest

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt 

RUN pip3 install flask[async]

EXPOSE 5001

CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5001"]
