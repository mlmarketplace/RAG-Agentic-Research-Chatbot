import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Agentic Research Support Chatbot",
    page_icon="Research",
    layout="centered"
)

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------------------
# LOAD VECTOR STORE (CACHED)
# -----------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------------------
# LOAD LLM (CACHED)
# -----------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

llm = load_llm()

# -----------------------------------------
# PROMPT
# -----------------------------------------
prompt = ChatPromptTemplate.from_template("""
You are an Agentic Research Support Assistant for Applied AI Engineering teams.

Your role is to support researchers by grounding every answer strictly in the
provided document context. You do NOT rely on prior knowledge or assumptions.

---------------------
RULES OF OPERATION
---------------------

1. Use ONLY the supplied context to construct your answer.
2. If the answer is missing, incomplete, or ambiguous, respond exactly:
   "I’m not sure based on current documents."
3. Do NOT infer, generalize, or fill gaps using outside knowledge.
4. Do NOT fabricate:
   - methods
   - results
   - citations
   - definitions not explicitly present
5. If multiple documents disagree, summarize the differences without resolving them.
6. If the question is broader than the material, answer only the portion supported.
7. Never present speculation as fact.

---------------------
RESPONSE STYLE
---------------------

- Be precise, neutral, and technical.
- Avoid conversational language.
- Prefer extraction and synthesis over explanation.
- Keep answers concise but complete.
- For short and simple questions, use bullet points, limit to 8-10 bullet points


---------------------
CITATION REQUIREMENT
---------------------

For every key statement, cite the supporting passage using:
[Source: <document_name or metadata>]

If no citation can be provided, you must refuse.

---------------------
CONTEXT
---------------------
{context}

---------------------
RESEARCH QUESTION
---------------------
{question}

---------------------
OUTPUT FORMAT
---------------------

Answer:
<grounded response>

Sources:
- <document + section reference>
- <document + section reference>

If insufficient evidence exists, return only the refusal sentence.
""")

# -----------------------------------------
# SESSION STATE (CHAT MEMORY)
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------
# UI
# -----------------------------------------
st.title("Agentic Research Support Assistant")

st.markdown(
    "Ask research-related questions about internal documentation, "
    "technical guidelines, experiment logs, policies, or knowledge base materials. "
    "Responses are grounded strictly in approved documents."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------
# CHAT INPUT
# -----------------------------------------
user_question = st.chat_input("Ask an Agentic AI related question...")

if user_question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching documents..."):
        # Retrieve relevant docs
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # LLM response
        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=user_question
            )
        )

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })
