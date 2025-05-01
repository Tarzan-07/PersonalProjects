import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA

# Setup streamlit UI 
print("Starting Streamlit app...")
st.title("Rag system with Deepseek R1 and Ollama")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF file   
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split the documents into chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # initialze the embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # Initialize the LLM and chain
    llm = OllamaLLM(model="deepseek-r1:latest")
    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:  
    """

    QA_prompt = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_prompt)

    combine_docs_chain = create_stuff_documents_chain(llm=llm_chain, prompt=QA_prompt, document_variable_name="context")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        with st.spinner("Searching..."):
            result = qa(user_input)
            st.write(result["result"])