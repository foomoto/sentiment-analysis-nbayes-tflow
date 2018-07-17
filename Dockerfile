FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y git python3.6 python3-pip python3-dev build-essential libxml2-dev libxslt1-dev nginx
RUN git clone https://github.com/foomoto/sentiment-analysis-nbayes-tflow.git
WORKDIR sentiment-analysis-nbayes-tflow
RUN git pull
RUN pip3 install virtualenv
RUN virtualenv venv
RUN /bin/bash -c "source venv/bin/activate"
RUN pip3 install -r requirements.txt
CMD uwsgi --http 0.0.0.0:80 --wsgi-file www/web.py --callable app --processes 40 --threads 4 --stats 127.0.0.1:9191  > log_wsgi.out 2>&1&