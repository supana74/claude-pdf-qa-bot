import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import boto3

st.set_page_config(page_title="Claude PDF Bot", layout="wide")
st.title("📄 Claude PDF Q&A Bot using AWS Bedrock + LangChain")

# Upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Prompt template
prompt_template = PromptTemplate.from_template("""
You are Claude, an AI assistant helping a user understand a document.
Answer the question below using only the context provided from the PDF.

If the answer is not found, say “Answer not found in the PDF context.”

Context:
{context}

Question:
{question}

Answer:
""")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    # Load and chunk PDF
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed using Amazon Titan model
    embedding = BedrockEmbeddings(
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v1"
    )
    vectorstore = FAISS.from_documents(chunks, embedding)

    # Load Claude from Bedrock
    llm = BedrockChat(
        region_name="us-east-1",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    # Build Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # User input
    question = st.text_input("🔍 Ask a question about the PDF:")
    if st.button("Get Answer") and question:
        result = qa_chain({"query": question})
        st.success(result["result"])

        with st.expander("📄 Source context"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)

    # Optional: Summarize document
    if st.button("📘 Summarize Document"):
        docs = vectorstore.similarity_search("Summarize the document", k=8)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = prompt_template.format(context=context, question="Summarize this PDF.")
        summary = llm.invoke(prompt)
        st.info(summary.content)
