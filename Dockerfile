FROM ubuntu:latest
EXPOSE 80
RUN apt-get update -y
RUN apt-get install -y git python3.6 python3-pip python3-dev build-essential libxml2-dev libxslt1-dev
RUN git clone https://github.com/foomoto/sentiment-analysis-nbayes-tflow.git
WORKDIR sentiment-analysis-nbayes-tflow
RUN git pull
RUN pip3 install virtualenv
RUN virtualenv venv
RUN /bin/bash -c "source venv/bin/activate"
RUN pip3 install -r requirements.txt
CMD ./start.sh && /bin/bash