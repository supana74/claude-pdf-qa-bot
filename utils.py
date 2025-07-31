from langchain.prompts import PromptTemplate

def get_prompt_template():
    return PromptTemplate.from_template("""
You are Claude, an AI assistant helping a user understand a document.
Answer the question below using only the context provided from the PDF.

If the answer is not found, say “Answer not found in the PDF context.”

Context:
{context}

Question:
{question}

Answer:
""")
