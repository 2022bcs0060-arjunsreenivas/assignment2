FROM python:3.11-slim

WORKDIR /app

COPY app/app.py .
COPY requirements_docker.txt .
COPY output/model.pkl .

RUN pip install --upgrade pip
RUN pip install -r requirements_docker.txt

EXPOSE 8000

CMD [ "uvicorn","app:app","--host","0.0.0.0","--port","8000" ]