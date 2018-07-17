#!/usr/bin/env bash
PYTHONIOENCODING=utf-8
PYTHONPATH=~/octopus-brain
git pull
pkill uwsgi
source venv/bin/activate
uwsgi --http 127.0.0.1:8080 --wsgi-file www/web.py --callable app --processes 40 --threads 4 --stats 127.0.0.1:9191
#uwsgi --ini wsgi.ini
#> log_wsgi.out 2>&1&