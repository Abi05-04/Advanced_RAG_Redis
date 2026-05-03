# 🚀 Advanced RAG with Reranker & TTL Cache using Redis
 

## 📌 Project Overview  
This project implements an Advanced Retrieval-Augmented Generation (RAG) system designed to deliver accurate, fast, and context-aware answers.

It enhances traditional RAG by integrating:
- Semantic Retrieval using FAISS  
- Reranking using CrossEncoder  
- Redis Semantic Cache with TTL for faster responses  
- Local LLM (TinyLlama via Ollama) for answer generation  

The system efficiently retrieves, refines, and generates high-quality responses while reducing latency through intelligent caching.

---

## 🧠 Architecture Overview  

User Query  
↓  
FAISS Vector Search (Top-K Retrieval)  
↓  
CrossEncoder Reranker  
↓  
Filtered Relevant Context  
↓  
TinyLlama (Ollama LLM)  
↓  
Final Answer  
↓  
Redis Cache (with TTL)  

---

## 🔄 Workflow  

1. User enters a query via Streamlit UI  
2. FAISS retrieves top relevant document chunks  
3. CrossEncoder reranks results based on relevance  
4. Best context is passed to TinyLlama  
5. LLM generates final answer  
6. Redis caches the response with TTL  
7. Repeated queries are served instantly from cache  

---

## 🚀 Key Features  

- Semantic document retrieval using FAISS  
- Intelligent reranking using CrossEncoder  
- Fast response using Redis Semantic Cache  
- TTL support for automatic cache expiration  
- Local LLM integration (TinyLlama via Ollama)  
- Interactive Streamlit UI  
- Wikipedia-based document source  

---

## 🛠️ Tech Stack  

- Python  
- Streamlit  
- LangChain  
- Redis  
- FAISS  
- HuggingFace Embeddings  
- Sentence Transformers  
- Ollama (TinyLlama)  

---

## 📂 Project Structure  

Advanced-RAG-Redis/  
│── Reranker_web.py  
│── requirements.txt  
│── README.md  

---

## ⚙️ Installation & Setup  

### 1. Clone Repository  
git clone https://github.com/yourusername/Advanced-RAG-Redis.git  
cd Advanced-RAG-Redis  

### 2. Install Dependencies  
pip install -r req.txt  

### 3. Start Ollama  
ollama serve  

### 4. Pull Model  
ollama pull phi3  

### 5. Run Application  
streamlit run app.py    

---
# Screenshots
<img width="1920" height="1080" alt="Screenshot (161)" src="https://github.com/user-attachments/assets/e212666e-4a7c-4944-9a8a-3673a428494b" />

<img width="1920" height="1080" alt="Screenshot (162)" src="https://github.com/user-attachments/assets/70ee509c-ecc4-4aec-b508-3f5c0bfecfa3" />

<img width="1920" height="1080" alt="Screenshot (163)" src="https://github.com/user-attachments/assets/bda7c40d-e7d4-44fe-860f-fa085dc868e8" />



## 🖥️ Usage  

- Enter a query in the Streamlit interface  
- System retrieves relevant documents  
- Reranker improves context selection  
- LLM generates an accurate answer  
- Cached responses improve speed for repeated queries  

---

## 📊 Example Use Cases  

- Document Question Answering  
- Knowledge Retrieval Systems  
- AI Assistants  
- Research Support Tools  

---

## ⚡ Performance Highlights  

- Faster responses with Redis caching  
- Improved accuracy using reranker  
- Reduced LLM calls using semantic cache  
- Scalable and efficient architecture  

---

## 🔮 Future Enhancements  

- Multi-document support  
- Real-time data ingestion  
- Docker & Kubernetes deployment  
- Advanced reranking models  
- Chat history integration  

---

## ⚠️ Troubleshooting  

Error:  
lookup registry.ollama.ai: no such host  

Solution:  
- Check internet connection  
- Change DNS to 8.8.8.8  
- Disable VPN/Proxy  
- Try mobile hotspot  

---

## 🤝 Contributing  
Contributions are welcome! Feel free to fork and submit a pull request.  

---

## 📜 License  
MIT License  

---
## 👩‍💻 Author  
Abinaya Duraisamy

## ⭐ If you like this project, give it a star!
