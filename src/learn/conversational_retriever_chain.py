from openai import OpenAI
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval_tools.retrieval_utils import *
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def conversational_retrieval_chain(url):
    llm = ChatOpenAI()
    # First we need a prompt that we can pass into an LLM to generate this search query
    docs = docs_loaded_from_web(url)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever = build_whole_text_recursive_faiss_retriever(docs)

    print(f'number of documents after retrieving from {url} : {len(retriever.get_relevant_documents(""))}')
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response_docs = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(f' first retriever_chain returns when invoked a {type(response_docs)} of  {len(response_docs)} Documents from the vector store')

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    print(f'combining retrieval chain and document chain gives a best output ( we now look from the retrieved documents )')
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response_combined_chain = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(f'response_combined_chain.keys() : {response_combined_chain.keys()}')
    print(f'''response_combined_chain['chat_history'] : {response_combined_chain['chat_history']}''')
    print(f'''response_combined_chain['input'] : {response_combined_chain['input']}''')
    print(f'''response_combined_chain['answer'] : {response_combined_chain['answer']}''')

if __name__ == '__main__':
    conversational_retrieval_chain(url= "https://docs.smith.langchain.com/overview")