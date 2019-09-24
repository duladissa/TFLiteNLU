FROM python:3.5

RUN mkdir /app
WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt
