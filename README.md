# 🤖 AI & Natural Language Processing Projects

This repository contains two innovative Streamlit web applications developed for a Natural Language Processing and AI course. These applications showcase real-world implementations of Retrieval-Augmented Generation (RAG), Semantic Search, Vision-Language Models (VLM), and Text-to-Speech (TTS).

---

## 1. 🖼️ Interactive Image-to-Story App (`Interactive_APP.py`)

An interactive AI-powered application that takes an image as input and transforms it into an audible creative story.

### ✨ Features
- **Image Upload & URL Support**: Users can upload a local image (PNG/JPG) or provide an image URL.
- **Image Captioning**: Uses the `Salesforce/blip-image-captioning-base` Vision-Language Model to automatically describe the contents of the image.
- **Story Generation**: Connects to the **OpenAI API (`gpt-4o-mini`)** to write a creative short story based on the generated image caption.
- **Text-to-Speech (TTS)**: Synthesizes the generated story into realistic human speech using the **Kokoro TTS pipeline**. 

### 🚀 Usage
To run the interactive app:
```bash
streamlit run Interactive_APP.py
```
> **Note:** You will need to provide your OpenAI API credentials (`OPENAI_KEY` and `OPENAI_ORG`) within the UI or integrate them into the script to enable the story generation functionality.

---

## 2. 🔎 Semantic Search Web Application (`RAG.py`)

A full-fledged Retrieval-Augmented Generation (RAG) and Semantic Search pipeline capable of fetching relevant documents and answering user questions explicitly from the retrieved context.

### ✨ Features
- **Sparse Dense Retrieval**: Uses **SPLADE** (`naver/splade-cocondenser-ensembledistil`) via PyTorch and Transformers to map document chunks into a sparse vector space.
- **Inverted Index Engine**: Builds an inverted index of the document corpus, enabling fast and scalable document retrieval via vector dot-products.
- **Context-Aware Question Answering**: Retrieves the Top-3 most relevant documents for a query and uses an external API hosting **Gemma-3** to answer the question strictly based on the extracted evidence without hallucinating.
- **Environment Driven**: Connects to the LLM backend via endpoints configured securely using a `.env` file for high portability.

### 🚀 Usage
To run the semantic search RAG:
```bash
streamlit run RAG.py
```
> **Note:** Ensure a `.env` file is present in the root directory defining `API_URL` and `API_TOKEN`. Required libraries include `streamlit`, `torch`, `transformers`, `requests`, and `python-dotenv`.

---

### ⚙️ Technologies Used
- Streamlit
- PyTorch & HuggingFace Transformers
- OpenAI API
- Kokoro TTS
- SPLADE (Sparse Vector Expansion)
- Python `requests` & `python-dotenv`
