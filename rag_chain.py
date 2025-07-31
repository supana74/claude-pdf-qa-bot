from langchain.chat_models import BedrockChat
from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore):
    llm = BedrockChat(
        region_name="us-east-1",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

def get_summary(vectorstore, prompt_template):
    llm = BedrockChat(
        region_name="us-east-1",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    docs = vectorstore.similarity_search("Summarize the document", k=8)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(context=context, question="Summarize this PDF.")
    return llm.invoke(prompt).content
