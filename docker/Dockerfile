FROM langchain/langchain:0.1.0

WORKDIR /app

COPY requirements.txt .
COPY retriever.py .
COPY db_builder.py .

RUN pip install -r requirements.txt

COPY database db