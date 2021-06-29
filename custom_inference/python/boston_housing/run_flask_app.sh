export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export FLASK_APP=server.app
export FLASK_ENV=development
cd src && python -m flask run --host 0.0.0.0 --port 8080