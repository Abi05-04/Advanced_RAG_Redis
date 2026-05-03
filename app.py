import os

# Fix CPU / tokenizer warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="🧠 Semantic QA Reranker",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    # 🔥 Use phi3 (lightweight & stable)
    llm = ChatOllama(
        model="phi3",
        base_url="http://localhost:11434" 
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return llm, embeddings, reranker


# -----------------------------
# RETRIEVER
# -----------------------------
@st.cache_resource
def get_retriever(query, embeddings):
    loader = WikipediaLoader(query=query, load_max_docs=5)
    docs = loader.load()

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 10})


# -----------------------------
# UI
# -----------------------------
st.title("🧠 Semantic Question Answering with Reranker")

st.markdown("""
Ask any question.

This app uses:
- Wikipedia Retrieval  
- FAISS Vector Search  
- CrossEncoder Reranking  
- Ollama Local LLM  
""")

llm, embeddings, reranker = load_models()

query = st.text_input("Enter your question:")

if st.button("Search & Answer") and query:

    try:
        # Step 1: Retrieve documents
        with st.spinner("🔍 Retrieving documents..."):
            retriever = get_retriever(query, embeddings)

            if retriever is None:
                st.error("No documents found. Try another query.")
                st.stop()

            docs = retriever.invoke(query)

        # Step 2: Rerank
        with st.spinner("📊 Reranking results..."):
            pairs = [(query, d.page_content) for d in docs]
            scores = reranker.predict(pairs)

            ranked = sorted(
                zip(docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [doc for doc, _ in ranked[:5]]

        # Step 3: Build context
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Step 4: Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer only from the given context.\n"
             "If not found, say: I don't know.\n\nContext:\n{context}"),
            ("user", "{question}")
        ])

        # Step 5: Generate answer
        with st.spinner("🤖 Generating answer..."):
            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": query
            })

        # OUTPUT
        st.subheader("💡 Final Answer")
        st.success(response.content)

        # Show retrieved chunks
        with st.expander("📄 View Retrieved Chunks"):
            for i, doc in enumerate(top_docs, 1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)

    except Exception as e:
        st.error(f"Error: {e}")