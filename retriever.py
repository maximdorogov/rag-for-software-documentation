import argparse
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

from langchain_mistralai import ChatMistralAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='',
            add_help=True,
        )

    parser.add_argument('--cloud',
                        help=('Process on cloud'),
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    RAG_MODEL_NAME = 'HuggingFaceTB/SmolLM-135M'
    EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    DB_PATH = './database/chroma'

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(embedding_function=embedding, persist_directory=DB_PATH)

    if args.cloud:
        os.environ["MISTRAL_API_KEY"] = 'zRzdF88M3yk3EYRurBVJk0b3zbTlI9mt'
        llm = ChatMistralAI(model="mistral-large-latest")
    else:
        llm = HuggingFacePipeline.from_model_id(
            model_id=RAG_MODEL_NAME,
            pipeline_kwargs={"temperature": 0, "max_new_tokens": 20},
            task='text-generation'
        )

    # chat_llm = ChatHuggingFace(llm=llm)

    # embedding = HuggingFaceHubEmbeddings(
    #     model=RAG_MODEL_NAME,
    #     huggingfacehub_api_token='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'
    # )

    # llm = HuggingFaceEndpoint(
    #     repo_id=RAG_MODEL_NAME,
    #     hugginfacebub_api_token='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',
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type='refine',
        retriever=vector_db.as_retriever(search_kwargs={'k': 3})
    )

    question = 'What are all AWS regions where SageMaker is available?'
    result = qa_chain.invoke(question)
    print(result['result'])

    # docs = vector_db.similarity_search(question, k=3)
    # for doc in docs:
    #     print(doc.page_content)
