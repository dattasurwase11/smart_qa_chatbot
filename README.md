# 📄💬 Smart PDF Q&A Assistant

A modern, industry-ready chatbot that answers natural-language questions about your PDF documents using **Hugging Face LLMs** and **Retrieval-Augmented Generation (RAG)**.  
Built with [Streamlit](https://streamlit.io/) for a beautiful, interactive UI.

---

## 🚀 Features

- **PDF Upload:** Easily upload any PDF document.
- **RAG Pipeline:** Uses vector search (FAISS + BGE embeddings) to retrieve relevant context from your PDF and augments the LLM prompt for accurate, context-based answers.
- **Hugging Face LLM:** Uses FLAN-T5 for high-quality, open-source language generation.
- **Example Questions:** Get inspired with sample queries.
- **Clear & About Buttons:** Reset the chat or learn about the bot.
- **Modern UI:** Clean, colorful, and responsive design with custom CSS.
- **Efficient:** Uses caching for fast repeated queries and model loading.
- **Local & Private:** All processing happens locally—your data stays with you.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [LangChain](https://python.langchain.com/) — RAG orchestration
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — LLM and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Vector database
- [PyPDF](https://pypdf.readthedocs.io/) — PDF parsing

---

## 📦 Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/smart-pdf-qa-chatbot.git
    cd smart-pdf-qa-chatbot
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

1. **Run the app:**
    ```bash
    streamlit run huggingface_LLM_Rag.py
    ```

2. **Open your browser:**  
   Go to [http://localhost:8501](http://localhost:8501)

3. **Upload a PDF and ask questions!**

---

## 💡 Example Questions

- *What is the main topic of this document?*
- *Summarize the key findings in this PDF.*
- *List the main sections of this document.*

---

## 📝 How It Works

1. **Upload PDF:** The app extracts and splits the text into chunks.
2. **Embed & Store:** Each chunk is embedded using BGE embeddings and stored in a FAISS vector store.
3. **Ask a Question:** Your question is embedded and used to retrieve the most relevant chunks.
4. **LLM Answer:** The context and your question are sent to the FLAN-T5 model, which generates an answer based only on your document.

---

## 🔒 Privacy

All processing is done locally. Your PDFs and questions are **never sent to any external server**.

---

## 🤝 Credits

- [Hugging Face](https://huggingface.co/)
- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyPDF](https://pypdf.readthedocs.io/)

---

## 📜 License

MIT License

---

*Built with ❤️ for private, intelligent document Q&A.*
