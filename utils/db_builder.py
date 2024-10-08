from typing import List
import argparse
import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def list_docs(path: str) -> List[str]:
    """
    Returns absolute path of each documents from the dataset
    """
    return [
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))]

def load_markdown_docs(paths: List[str]) -> List[Document]:
    """
    Loads markdown documents with UnstructuredMarkdownLoader
    """
    return [UnstructuredMarkdownLoader(f).load()[0] for f in paths]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Script to build the chroma vector database',
            add_help=True,
        )
    parser.add_argument('-d', '--input_data_path',
                        help=('Dataset path folder'),
                        type=str,
                        required=True)
    parser.add_argument('-cz', '--chunk_size',
                        help=('Document chunk size after splitting'),
                        type=int,
                        required=False,
                        default=1000)
    parser.add_argument('-co', '--chunk_overlap',
                        help=('Overlap between chunks after splitting'),
                        type=int,
                        required=False,
                        default=90)
    parser.add_argument('-db', '--db_path',
                        help=('Database output folder'),
                        type=str,
                        required=True)
    parser.add_argument('-m', '--embedding_model_name',
                        help=('hugginface model for embedding generation'
                        ),
                        type=str,
                        required=False,
                        default='sentence-transformers/all-mpnet-base-v2'
                        )
    parser.add_argument('-m', '--embedding_model_path',
                        help=('hugginface model path'
                        ),
                        type=str,
                        required=False,
                        default='./all-mpnet-base-v2'
                        )
    args = parser.parse_args()
    
    # create output folder for db
    os.makedirs(args.db_path, exist_ok=True)

    embedding = HuggingFaceEmbeddings(
        model_name=args.embedding_model_name,
        cache_folder=args.embedding_model_path,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    document_paths = list_docs(args.input_data_path)
    docs = load_markdown_docs(paths=document_paths)
    splits = splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=args.db_path
    )