import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap=0)
    chunks = text_splitter.create_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    template = """
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        Human: {question}
        AI:
        """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    retriever= vectorstore.as_retriever(search_kwargs=dict(k=3))
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )

    return chain;

def handle_use_input(user_question):
    response = st.session_state.conversation({'query': user_question})
    
    
    st.session_state.chat_history.append({"user": user_question, "bot": response["result"]})

    for exchange in st.session_state.chat_history:
        message = st.chat_message("Human")
        message.write(exchange["user"])
        message = st.chat_message("AI")
        message.write(exchange["bot"])

def main():
    load_dotenv()
    st.set_page_config(page_title="git assistant") 


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown("Made in ![this is an image link](https://i.imgur.com/iIOA6kU.png) , with :heart: by ***ChetanKumar***.")

    st.header(':blue[Your Git Assistant] :male-teacher:', divider='rainbow')
    user_question = st.chat_input("Shoot Your Questions Below:")

    if "raw_text" not in st.session_state:
        st.session_state.raw_text = get_pdf_text(r'C:\Users\ChetanKumar\workspace\git_bot\progit.pdf')
        
    
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = get_text_chunks([st.session_state.raw_text])
    

    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore(st.session_state.text_chunks)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

    if user_question:
        handle_use_input(user_question)

    
    
if __name__ == '__main__':
    main()