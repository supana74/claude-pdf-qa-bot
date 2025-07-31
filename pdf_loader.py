import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding = BedrockEmbeddings(
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v1"
    )
    vectorstore = FAISS.from_documents(chunks, embedding)

    return docs, vectorstore
