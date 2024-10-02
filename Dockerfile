FROM python:3.9.6

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 7860

ENTRYPOINT ["python", "main.py"]

CMD ["--type", "default"]

