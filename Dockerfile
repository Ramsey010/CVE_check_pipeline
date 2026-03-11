FROM python:3.8-slim
LABEL maintainer="Ramsey010"
LABEL project="scan"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Data_format.py .
CMD ["python", "Data_format.py","--help"]
