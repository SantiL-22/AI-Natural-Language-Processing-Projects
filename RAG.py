# ============================================================
# Student Assignment: Semantic Search Web Application
# ============================================================

import streamlit as st
import requests
import torch
import heapq
from math import sqrt
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Semantic Search with SPLADE", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: white; padding: 10px;'>
        🔎 Semantic Search Web Application
    </h1>
    """,
    unsafe_allow_html=True
)

# =========================
# DEVICE + MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
mlm = AutoModelForMaskedLM.from_pretrained(
    "naver/splade-cocondenser-ensembledistil"
).to(device)
mlm.eval()

VERBOSE = False

# =========================
# SPLADE FUNCTIONS
# =========================
@torch.no_grad()
def splade_pool(input_ids, attention_mask):
    out = mlm(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    act = torch.log1p(torch.relu(logits))
    mask = attention_mask.unsqueeze(-1)
    act = act * mask
    pooled = act.max(dim=1).values
    return pooled

def doc_to_chunks(title, text, max_len=480, stride=416):
    s = (title.strip() + " " + text.strip()).strip()
    enc = tok(s, return_tensors="pt", truncation=False, add_special_tokens=True)
    ids = enc["input_ids"][0]
    att = torch.ones_like(ids)

    total_len = len(ids)
    if total_len == 0:
        return []

    starts, start = [], 0
    while start < total_len:
        starts.append(start)
        end = min(start + max_len, total_len)
        if end == total_len:
            break
        start += stride

    chunks = []
    for start in starts:
        end = min(start + max_len, total_len)
        cid = ids[start:end]
        cat = att[start:end]
        chunks.append({
            "input_ids": torch.nn.functional.pad(cid, (0, max_len - len(cid))),
            "attention_mask": torch.nn.functional.pad(cat, (0, max_len - len(cat)))
        })
    return chunks

@torch.no_grad()
def splade_document(title, text, topk_keep=100, verbose=False):
    chunks = doc_to_chunks(title, text)
    if not chunks:
        return {}

    batch_ids = torch.stack([c["input_ids"] for c in chunks]).to(device)
    batch_att = torch.stack([c["attention_mask"] for c in chunks]).to(device)

    per_chunk = splade_pool(batch_ids, batch_att)
    doc_vec = per_chunk.max(dim=0).values

    vals, idx = torch.topk(doc_vec, k=min(topk_keep, doc_vec.numel()))
    vals, idx = vals[vals > 0], idx[vals > 0]
    return {int(i): float(v) for i, v in zip(idx, vals)}

@torch.no_grad()
def splade_query(query, topk_keep=64):
    enc = tok(query, return_tensors="pt", truncation=True, max_length=64).to(device)
    qv = splade_pool(enc["input_ids"], enc["attention_mask"])[0]
    vals, idx = torch.topk(qv, k=min(topk_keep, qv.numel()))
    vals, idx = vals[vals > 0], idx[vals > 0]
    return {int(i): float(v) for i, v in zip(idx, vals)}

def build_inverted_index(corpus, topk_keep=100):
    inv = defaultdict(list)
    doc_norm = {}

    for doc_id, d in tqdm(corpus.items(), desc="Indexing documents"):
        vec = splade_document(d.get("title", ""), d.get("text", ""), topk_keep)
        norm = sqrt(sum(v*v for v in vec.values())) or 1.0
        doc_norm[doc_id] = norm
        for tid, w in vec.items():
            inv[tid].append((doc_id, w))

    return inv, doc_norm

def retrieve(inv_index, doc_norm, query, topk=3, normalize=True):
    q = splade_query(query)
    qnorm = sqrt(sum(v*v for v in q.values())) or 1.0
    scores = defaultdict(float)

    for tid, qw in q.items():
        for doc_id, dw in inv_index.get(tid, []):
            scores[doc_id] += qw * dw

    if normalize:
        for doc_id in scores:
            scores[doc_id] /= (doc_norm[doc_id] * qnorm)

    return heapq.nlargest(topk, scores.items(), key=lambda x: x[1])

# =========================
# CORPUS + QUERIES
# =========================
corpus_test = {
    "doc1":  "Machine learning models learn patterns from data and features",
    "doc2":  "Artificial intelligence includes reasoning planning and perception in machines",
    "doc3":  "Data science combines statistics programming and domain knowledge for insights",
    "doc4":  "Deep learning uses neural networks with many layers for vision",
    "doc5":  "TF IDF weighs frequent terms and downweights common corpus words",
    "doc6":  "BM25 improves tf idf with saturation and length normalization",
    "doc7":  "Cosine similarity compares vector directions for document ranking tasks",
    "doc8":  "Information retrieval finds relevant documents for a user query quickly",
    "doc9":  "Retrieval augmented generation combines search with a language model answer",
    "doc10": "SPLADE produces sparse expansions using masked language modeling logits",
    "doc11": "Indexing builds an inverted index mapping terms to document postings",
    "doc12": "Tokenization splits text into tokens sometimes using subword pieces",
    "doc13": "Stopwords like the and is can be removed to reduce noise",
    "doc14": "Stemming reduces words to roots like compute computing computed",
    "doc15": "Lemmatization maps words to dictionary forms using part of speech",
    "doc16": "Precision measures fraction of retrieved documents that are relevant",
    "doc17": "Recall measures fraction of relevant documents that are retrieved",
    "doc18": "Mean average precision summarizes ranking quality across many queries",
    "doc19": "Vector databases store embeddings for fast nearest neighbor search",
    "doc20": "Dense retrieval uses neural embeddings rather than exact term matching",
    "doc21": "Sparse retrieval relies on term weights and inverted index structures",
    "doc22": "Hybrid search combines sparse and dense signals for better recall",
    "doc23": "Query expansion adds related terms to improve recall in search",
    "doc24": "Synonyms can help retrieval but may introduce ambiguity in results",
    "doc25": "Ranking functions score documents based on term matches and weights",
    "doc26": "Chunking splits long documents into windows to fit model limits",
    "doc27": "Overlap stride keeps context continuity between adjacent text chunks",
    "doc28": "Normalization divides by vector norms to compute cosine style scores",
    "doc29": "Evaluation needs labeled relevance judgments from human assessors",
    "doc30": "A chatbot answers questions by retrieving evidence and generating text",
    "doc31": "Clinical notes require careful privacy handling and access control",
    "doc32": "Pipelines log steps outputs and metadata for reproducible analysis",
    "doc33": "Docker containers package apps with dependencies for portable deployment",
    "doc34": "Shiny apps serve interactive R dashboards through a web browser",
    "doc35": "GPU acceleration speeds up neural inference for large transformer models",
    "doc36": "Caching model files avoids repeated downloads and improves startup time",
    "doc37": "Attention mechanisms weigh token interactions in transformer encoders",
    "doc38": "Masked language modeling predicts missing tokens to learn representations",
    "doc39": "Distillation trains smaller models to mimic larger teacher outputs",
    "doc40": "Out of vocabulary terms may be split into multiple subword tokens",
    "doc41": "Relevance feedback lets users refine results and correct mismatches",
    "doc42": "Inverted index stores postings lists for each term in vocabulary",
    "doc43": "Document length affects BM25 scoring through normalization parameters",
    "doc44": "Term frequency saturation prevents long documents dominating rankings",
    "doc45": "RAG responses should cite sources and avoid hallucinating unsupported facts",
    "doc46": "Paella and tortilla are popular Spanish foods often served at lunch",
    "doc47": "Pizza and pasta are common Italian dishes with many regional variants",
    "doc48": "Running and cycling improve cardio fitness and overall health benefits",
    "doc49": "San Sebastian is known for pintxos beaches and culinary culture",
    "doc50": "Statistics includes hypothesis testing confidence intervals and p values",
}

queries_test = [
    "What is the difference between TF IDF and BM25 scoring?",
    "How does SPLADE create sparse query expansion terms?",
    "Why do we use cosine similarity when ranking documents?",
    "What is retrieval augmented generation and why is it useful?",
    "How do chunking and overlap stride help long document retrieval?",
    "What is an inverted index and what does it store?",
    "Explain precision and recall in information retrieval evaluation",
    "How do dense, sparse, and hybrid search differ in practice?",
    "What are stopwords stemming and lemmatization used for?",
    "What are typical Spanish and Italian foods mentioned in the corpus?",
]

documents_example_test = dict()
for c in corpus_test:
    d_ = {'text':c , 'text':corpus_test[c] }
    documents_example_test[c] = d_   


#inv_index_ex_test, doc_norm_ex_test = build_inverted_index(documents_example_test, topk_keep=120)

# =========================
# INDEX (CACHEADO)
# =========================
@st.cache_resource
def load_index():
    return build_inverted_index(documents_example_test, topk_keep=120)

inv_index_ex_test, doc_norm_ex_test = load_index()

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a helpful assistant. You will receive a set of relevant documents and a user question.

Instructions:

1. Answer the question using ONLY the information in the provided documents.
2. Do NOT use any external knowledge and DO NOT guess. Never invent facts. 
3. If the documents contain enough information to answer the question exactly, answer clearly and concisely, citing the relevant documents inline using their IDs, e.g., [doc1], [doc2].
4. If the documents do NOT contain enough information to answer the question exactly, you MUST NOT try to fill in missing information and respond to what the answere is.
5. In that case (4), respond exactly as follows:
   "I can't answer your question explicitly with the provided documents. However, the following documents contain information related to your query: [docX], [docY], ..."
6. Always list all documents that mention any relevant information to the question, even if partial. Do not provide any additional information beyond what is in the documents.

These are the relevant documents:
_INSERT_DOCUMENTS_HERE_
"""
# =========================
# Query input - menú desplegable
# =========================
query_mode = st.selectbox(
    "Select input mode",
    ["Test queries", "Custom query"]
)

if query_mode == "Test queries":
    qtext = st.selectbox("Select query", queries_test)
else:
    qtext = st.text_input("Enter your query", "")

# =========================
# Mostrar resultados y LLM Answer entre título y input
# =========================
if qtext:
    # Retrieval
    results = retrieve(inv_index_ex_test, doc_norm_ex_test, qtext, topk=3)
    relevant_content = {d: corpus_test[d] for d, _ in results}

    # Top-3 Retrieved Documents
    st.markdown("<h3 style='text-align: center;'>📄 Top-3 Retrieved Documents</h3>", unsafe_allow_html=True)
    for doc_id, score in results:
        st.markdown(f"<p style='text-align: center;'><b>{doc_id}</b> — `{score:.4f}`</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{corpus_test[doc_id]}</p>", unsafe_allow_html=True)

    # LLM Answer
    msgs = [
        {"role": "system",
         "content": SYSTEM_PROMPT.replace("_INSERT_DOCUMENTS_HERE_", str(relevant_content))},
        {"role": "user", "content": qtext}
    ]

    payload = {"model": "gemma3", "stream": False, "messages": msgs}

    response = requests.post(
        "http://[IP_ADDRESS]/api/chat",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=200
    )

    st.markdown("<h3 style='text-align: center;'>🤖 LLM Answer</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{response.json()['message']['content']}</p>", unsafe_allow_html=True)
