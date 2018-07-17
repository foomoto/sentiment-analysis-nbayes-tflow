FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3.6 python3-pip python3-dev build-essential libxml2-dev libxslt1-dev nginx
COPY ../AmazonReviews /app
WORKDIR /app/www
RUN pip3 install -r requirements.txt
ENTRYPOINT ["uwsgi"]
CMD ["--http", "0.0.0.0:80", "--wsgi-file", "www/web.py", "--callable", "app", "--processes", "10", "--threads", "2", "--stats", "0.0.0.0:9191"]