# Cloud based RAG for AWS SageMaker documentation

## Summary
This POC (Proof of concept) consist in a RAG (Retrieval Augmented Generation) system that is capable of:

1. Receive a user query, usually a question, about the relevant topic i.e. AWS SageMaker.
1. Return: 
    - An answer to the user question.
    - A list of reference documents from SageMaker documentation for further consulting.

> The main request (answer some questions about SageMaker) and also the optional (return reference documents from documentation) were implemented.

High level overview of the system architecture:

![title](images/high_level_diag.png)


The system is composed of a vector database and an inference worker, both integrated as microservices. The inference worker is a rest API that communicates with the user and is responsable of quering the vector database and excecuting the LLM based pipeline to deliver the requested answer.

![title](images/database_inference.png)
> Detailed overview of the inference pipeline.

The embedding extraction is made on a locally running model, in order to save some credits since this is project uses free models and the credits burn really fast if you want to encode the whole dataset.
The inference part is done using MistralAI, a third party API similar to OpenAI, so you need to go to [mistral.ai](https://mistral.ai/) and generate your free api key.

The system output follows this schema:

```python
@dataclass
class LLMResponse:
    docs: Set[str]
    answer: str
```


## System setup

### Requirements
[Docker](https://docs.docker.com/manuals/) (thats all!)

### Database build

Since this is a demo based on opensource/free tools in order to speed up the excecution the documents must be parsed and the database must be built before the first excecution. 

Download the dataset from [HERE]()

To build the database run:
```sh
docker build . -f Dockerfile

```
> You can also download the built database from [HERE]()

## Build and deploy

Download the embeddings extraction model from [HERE](). And place it in the root
this project.

Create a `.env` file and place it in the root of this project.
Your `.env` should look like this:

```
MISTRAL_API_KEY={your mistral api key}
EMBEDDING_MODEL_PATH=./all-mpnet-base-v2
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
INFERECE_MODEL=mistral-large-latest
```
Once the database is build place it in `/database` and run (from the root directory of this project):

```sh
docker compose up
```

this will bring the system up.

Once the system is up you can go to [http://localhost:80/docs](http://localhost:80/docs).

![title](images/fast_api_docs.png)

You can test the inference endpoint from there or try to use `curl`:

```sh
curl -X 'POST' \
  'http://localhost/inference_endpoint?message=How%20to%20check%20if%20an%20endpoint%20is%20KMS%20encrypted' \
  -H 'accept: application/json' \
  -d ''
```


## Next Steps

1. Implement an endpoint to update/delete documents from the existing database. The current database has been setted up to use persistent volumes so the only modification needed is in the `inference_worker` service.
2. Switch to OpenAI api for inference and embedding extraction.


