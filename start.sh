#! /bin/bash
PYTHONIOENCODING=utf-8
uwsgi --http 0.0.0.0:8080 --wsgi-file www/web.py --callable app --processes 40 --threads 4 --stats 127.0.0.1:9191  > log_wsgi.out 2>&1&