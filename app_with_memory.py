import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

# -----------------------------------------
# ENV
# -----------------------------------------
load_dotenv()

# print(f"\nOPENAI_API_KEY starts with"
#       f" {os.getenv('OPENAI_API_KEY')[:5]}")

st.set_page_config(
    page_title="RAG agentic Research Chatbot",
    page_icon="💼",
    layout="centered"
)

# -----------------------------------------
# CONSTANTS
# -----------------------------------------
VECTOR_DB_PATH = "hr_faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------------------
# LOAD VECTOR STORE
# -----------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------------------
# LOAD LLM
# -----------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)



llm = load_llm()

# -----------------------------------------
# PROMPT (MEMORY-AWARE)
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
# SESSION STATE
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------
# UI
# -----------------------------------------
st.title("💼 Agentic Research Chatbot (Memory Enabled)")
st.markdown(
    "Ask Agentic Research and Learning questions. Follow-up questions are supported."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------------
# CHAT INPUT
# -----------------------------------------
user_question = st.chat_input("Ask an Agentic AI research and learning related question...")

if user_question:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Thinking..."):
        # Build chat history string (limit to last 6 messages)
        history = st.session_state.messages[-6:]
        chat_history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        # Retrieve docs using current question only
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = llm.invoke(
            prompt.format_messages(
                chat_history=chat_history,
                context=context,
                question=user_question
            )
        )

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })
