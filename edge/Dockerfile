FROM base_image:latest

COPY . /app

WORKDIR /app

RUN pip3 install flask[async]

EXPOSE 5000

CMD ["flask", "run", "-p", "5000"]
