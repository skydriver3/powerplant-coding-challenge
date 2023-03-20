FROM python:3.10.9

ADD main.py .
ADD minimum.py .
ADD requirements.txt .

RUN pip install flask flask-restful numpy 

CMD [ "python", "./main.py"]