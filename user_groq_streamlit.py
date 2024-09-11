import os
import streamlit as st
import logging
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

# Caching vector store
@st.cache_resource
def load_vector_store():
    try:
        base_index_path = r"path\to\your\pdfs"
        num_batches = 3  # Update this if you have more batches

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = None

        for i in range(num_batches):
            index_path = f"{base_index_path}{i}_index"
            batch_vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            logging.info(f"FAISS index batch {i} loaded successfully from {index_path}")

            if vectorstore is None:
                vectorstore = batch_vectorstore
            else:
                vectorstore.merge_from(batch_vectorstore)

        return vectorstore
    except Exception as e:
        logging.error(f"Failed to load vector store batches: {e}")
        return None

# Caching RAG chain
@st.cache_resource
def create_rag_chain(_vectorstore):
    try:
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)

        retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        prompt_template = """You are an AI assistant tasked with providing accurate information based on the given context. Your goal is to extract relevant information and present it clearly.

        Context:
        {context}

        Question: {question}

        Instructions:
        1. Provide an answer based on the information in the given context.
        2. If the context contains relevant information, use it to construct a clear and informative answer.
        3. If the exact answer is not in the context but related information is present, provide that information and explain its relevance.
        4. Quote relevant parts of the context when appropriate, using quotation marks.
        5. If multiple sources provide relevant information, synthesize them into a coherent answer.

        Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for i, doc in enumerate(docs))

        rag_chain = (
            RunnableParallel(
                {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    except Exception as e:
        logging.error(f"Failed to create RAG chain: {e}")
        return None

# Function to auto-save chats
def auto_save_conversation():
    if "previous_chats" not in st.session_state:
        st.session_state.previous_chats = {}

    if len(st.session_state.messages) > 0:
        chat_title = st.session_state.messages[0]["content"][:30]
        st.session_state.previous_chats[chat_title] = st.session_state.messages

# UI Function to make the application more interactive and visually appealing
def main():
    st.set_page_config(page_title="Legal GPT Assistant", page_icon="pic.jpg", layout="wide")
    st.title("⚖️ Legal GPT Assistant")

    with st.sidebar:
        st.header("📂 Chat Management")

        if st.button("Start New Chat"):
            st.session_state.messages = []
            st.session_state.context = ""

        if "previous_chats" not in st.session_state:
            st.session_state.previous_chats = {}

        previous_chats = st.session_state.previous_chats
        if previous_chats:
            st.subheader("Previous Chats")
            for chat_title in previous_chats:
                if st.button(chat_title):
                    st.session_state.messages = previous_chats[chat_title]
                    st.session_state.context = st.session_state.messages[-1]['content']

    vectorstore = load_vector_store()
    if vectorstore is None:
        st.error("Failed to load vector store. Please check the logs for more information.")
        return

    rag_chain = create_rag_chain(vectorstore)
    if rag_chain is None:
        st.error("Failed to create RAG chain. Please check the logs for more information.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.context = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What would you like to know about legal matters?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Generating response..."):
                time.sleep(2)

                try:
                    combined_context = st.session_state.context + "\n\n" + prompt
                    response = rag_chain.invoke(combined_context)
                    full_response = response
                    message_placeholder.markdown(full_response)

                    with st.expander("View Source Documents"):
                        st.markdown(response)
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}\nPlease try asking your question again."
                    message_placeholder.error(error_message)
                    full_response = error_message

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.context = full_response

        auto_save_conversation()

    st.markdown("---")
    st.markdown("### Download Conversation")
    if st.button("Download"):
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(label="Download as .txt", data=conversation, file_name="conversation.txt")

if __name__ == "__main__":
    main()
