# Import libraries
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import textract

# Install sqlite3 module
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Define a function to generate response from uploaded files and query text
def generate_response(uploaded_files, openai_api_key, query_text):
    # Initialize an empty list to store documents
    documents = []
    # Loop through the uploaded files
    for uploaded_file in uploaded_files:
        # Parse the document based on its type
        if uploaded_file.type == 'application/pdf':
            # Get the file content as bytes and pass it to textract.process with extension argument
            document = textract.process(uploaded_file.getvalue(), extension='pdf', method='pdfminer').decode()
        elif uploaded_file.type == 'text/plain':
            # Read the file content as string
            document = uploaded_file.read().decode()
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Get the file content as bytes and pass it to textract.process with extension argument
            document = textract.process(uploaded_file.getvalue(), extension='docx').decode()
        else:
            # Skip unsupported file types
            continue
        # Append the document to the list of documents
        documents.append(document)
    # Split documents into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings using OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents using Chroma
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface from vectorstore
    retriever = db.as_retriever()
    # Create QA chain using RetrievalQA, OpenAI, and retriever
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    # Run the QA chain with the query text and return the response
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload (allow multiple files)
uploaded_files = st.file_uploader('Upload one or more articles', type=['pdf', 'txt', 'docx'], accept_multiple_files=True)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_files)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_files and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
