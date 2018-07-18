FROM ubuntu:latest
EXPOSE 80
RUN apt-get update -y
RUN apt-get install -y git python3.6 python3-pip python3-dev build-essential libxml2-dev libxslt1-dev nginx
COPY * /sentiment-analysis-nbayes-tflow/
WORKDIR /sentiment-analysis-nbayes-tflow
RUN pip3 install virtualenv
RUN virtualenv venv
RUN /bin/bash -c "source venv/bin/activate"
RUN pip3 install -r requirements.txt
CMD /start.sh