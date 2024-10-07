FROM langchain/langchain:0.1.0

WORKDIR /app
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./inference_worker .
COPY ./all-mpnet-base-v2 ./all-mpnet-base-v2
