from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader


def docs_loaded_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def build_splitted_documents(text_splitter,docs):
    documents = text_splitter.split_documents(docs)
    return documents


def build_faiss_vector_store(documents,embeddings):
    return FAISS.from_documents(documents, embeddings)


def build_retriever(vector_store):
    retriever = vector_store.as_retriever()
    return retriever


def build_whole_text_recursive_faiss_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter()
    documents = build_splitted_documents(text_splitter,docs)
    embeddings = OpenAIEmbeddings()
    vector_store = build_faiss_vector_store(documents,embeddings)
    retriever = build_retriever(vector_store)
    return retriever