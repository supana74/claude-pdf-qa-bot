import streamlit as st
from pdf_loader import load_and_split_pdf
from rag_chain import build_qa_chain, get_summary
from utils import get_prompt_template

st.set_page_config(page_title="Claude PDF Bot", layout="wide")
st.title("📄 Claude PDF Q&A Bot using AWS Bedrock + LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    docs, vectorstore = load_and_split_pdf(uploaded_file)
    qa_chain = build_qa_chain(vectorstore)
    prompt_template = get_prompt_template()

    question = st.text_input("🔍 Ask a question about the PDF:")
    if st.button("Get Answer") and question:
        result = qa_chain({"query": question})
        st.success(result["result"])

        with st.expander("See source context"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)

    if st.button("📘 Summarize Document"):
        summary = get_summary(vectorstore, prompt_template)
        st.info(summary)
