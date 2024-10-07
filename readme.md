# Cloud based RAG for AWS SageMaker documentation

This POC (Proof of concept) includes a dockerized RestAPI that will:

1. Receive a user query, usually a question, about the relevant topic i.e. AWS SageMaker.
1. Returns: 
    - An answer to the user question.
    - A list of reference documents from SageMaker documentation for further consulting.


High level overview of the system architecture:

![title](images/high_level_diag.png)

Inference worker:

## To do:

