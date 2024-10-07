import chromadb
import langchain_chroma.vectorstores
import langchain.chains.retrieval_qa
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Tuple, Set, Dict, Any
import os

class ChromaConnector:
    """
    Wrapper for all methods and initialization related to chromadb
    """
    def __init__(
        self,
        embedding_model_name:str,
        embedding_model_folder:str,
        db_host:str,
        db_port:int,
    ):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            cache_folder=embedding_model_folder,
        )
        self._chroma_client = chromadb.HttpClient(
            host=db_host, 
            port=db_port,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

    def init_vector_db(self) -> Chroma:
        return Chroma(
            client=self._chroma_client,
            embedding_function=self._embeddings)

class MistralAIRetriever:

    def __init__(
        self, 
        api_key: str, 
        vector_db: langchain_chroma.vectorstores,
        model_id: str = "mistral-large-latest",
    ) -> None:
        """
        Parameters
        ----------
        api_key: str

        vector_db: langchain_chroma.vectorstores

        model_id: str

        """
        self._vector_db = vector_db

        self.llm = ChatMistralAI(
            model=model_id,
            mistral_api_key=api_key
        )
        self.qa_chain = self._make_chain()

    def _make_chain(
        self, 
        chain_type: str = 'refine',
        top_k: int = 3
    ) -> langchain.chains.retrieval_qa:
        """
        Parameters
        ----------
        chain_type: str
            Allowed chain types are 
        top_k: int
            Amount of documents to be retrieved
        Return
        ------
        langchain.chains.retrieval_qa object
        """
        return RetrievalQA.from_chain_type(
            self.llm,
            chain_type=chain_type,
            retriever=self._vector_db.as_retriever(search_kwargs={'k': top_k}),
            return_source_documents=True,      
        )

    @staticmethod
    def _extract_sources(result: Dict[str, Any]) -> Set[str]:
        """
        Parameters
        ----------
        result: Dict[str, Any]
            Output from RetrievalQA.invoke() call.
        
        Return
        ------
        Names of documents as a Set of strings.
        """
        source_docs = [os.path.basename(doc.metadata['source']) 
                       for doc in result['source_documents']]
        return set(source_docs)

    def answer(self, question: str) -> Tuple[str, Set[str]]:
        """
        Parameters
        ----------
        question: str
            Question that will be asked to the llm.
        Return
        ------
        answer: Tuple[str, Set[str]]
            Containing the llm answer as string and a List of names of documents
            that contains detailed information about the answered topic.
        """
        result = self.qa_chain.invoke(question)
        source_docs = self._extract_sources(result=result)
        return result['result'], source_docs