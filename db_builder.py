import argparse
from typing import List
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
    TODO: Parametrize Loader type
    """
    return [UnstructuredMarkdownLoader(f).load()[0] for f in paths]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='',
            add_help=True,
        )

    parser.add_argument('-d', '--input_data_path',
                        help=('Path folder '),
                        type=str,
                        required=True)
    parser.add_argument('-cz', '--chunk_size',
                        help=('Path folder '),
                        type=int,
                        required=False,
                        default=1000)
    parser.add_argument('-co', '--chunk_overlap',
                        help=('Path folder '),
                        type=int,
                        required=False,
                        default=90)
    parser.add_argument('-db', '--db_path',
                        help=('Path folder '),
                        type=str,
                        required=True)
    parser.add_argument('-m', '--embedding_model_name',
                        help=('Identifier string that should be on the filename to '
                            'compare the base result with the one with the identifier'
                            ),
                        type=str,
                        required=False,
                        default='sentence-transformers/all-mpnet-base-v2'
                        )
    args = parser.parse_args()

    # create output folder for db
    os.makedirs(args.db_path, exist_ok=True)

    embedding = HuggingFaceEmbeddings(model_name=args.embedding_model_name)
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