 <div align="center">

# 🔎 Gemini Semantic Search

### Text Chunking • Gemini Embeddings • Cosine Similarity • Top-K Retrieval

A simple semantic search project built in **Google Colab** using the **Gemini Embedding API**.
 SDAIA Program for Developing AI Solution 


The notebook loads a text file, splits it into overlapping chunks, converts every chunk into a dense vector embedding, accepts a user question, and returns the **Top 3 most semantically relevant chunks** using cosine similarity.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-orange?logo=googlecolab)
![Gemini](https://img.shields.io/badge/Gemini-Embedding%20API-purple)
![Semantic Search](https://img.shields.io/badge/Search-Semantic-green)
![Similarity](https://img.shields.io/badge/Similarity-Cosine-blue)

</div>

---

## 📌 Overview

This project demonstrates a basic semantic information retrieval pipeline.

It:

- uploads a `.txt` file
- reads the document into Python
- splits the text into overlapping chunks
- generates embeddings using Gemini
- stores the vectors in a NumPy matrix
- accepts a natural-language query
- converts the query using the same embedding model
- calculates cosine similarity
- ranks all chunks by similarity
- returns the Top 3 most relevant chunks

---

## ✨ Features

- ✅ Google Colab compatible
- ✅ Gemini Embedding API
- ✅ Fixed-size text chunking
- ✅ 15% chunk overlap
- ✅ Dense vector embeddings
- ✅ Cosine similarity
- ✅ Semantic search
- ✅ Top-3 retrieval
- ✅ Similarity scores
- ✅ Natural-language questions
- ✅ Simple and beginner-friendly implementation

---

## 🧠 System Architecture

```text
TXT Document
     │
     ▼
Load Text
     │
     ▼
Fixed-Size Chunking
500 Characters
15% Overlap
     │
     ▼
Gemini Embedding Model
     │
     ▼
Dense Numerical Vectors
     │
     ▼
NumPy Embedding Matrix
     │
     ▼
User Question
     │
     ▼
Query Embedding
     │
     ▼
Cosine Similarity
     │
     ▼
Rank Chunks
     │
     ▼
Top 3 Results
```

---

## ⚙️ How It Works

### 1. Load the Text File

The notebook allows the user to upload a `.txt` file directly into Google Colab.

```python
uploaded = files.upload()
```

The uploaded file is then read using:

```python
with open(filename, "r", encoding="utf-8") as file:
    text = file.read()
```

---

### 2. Text Chunking

The document is split into fixed-size chunks of:

```text
500 characters
```

with:

```text
15% overlap
```

For a 500-character chunk, 15% overlap equals:

```text
75 characters
```

The step size becomes:

```text
500 - 75 = 425 characters
```

This overlap helps preserve context between neighboring chunks.

---

## 🤖 Gemini Embeddings

Each text chunk is converted into a numerical vector using:

```text
gemini-embedding-001
```

with:

```text
SEMANTIC_SIMILARITY
```

Example:

```python
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY"
    )
)
```

The same embedding function is used for both the document chunks and the user query.

This is important because both vectors must exist in the same semantic vector space.

---

## 🔐 Gemini API Key Setup

This project requires a Gemini API key.

### Step 1 — Create a Gemini API Key

Create an API key from Google AI Studio.

Do not paste your API key directly into the notebook if you plan to publish it on GitHub.

---

### Step 2 — Add the Key to Google Colab Secrets

In Google Colab:

1. Open your notebook.
2. Click the **🔑 Secrets** icon in the left sidebar.
3. Click **Add new secret**.
4. Use this name:

```text
GEMINI_API_KEY
```

5. Paste your Gemini API key as the value.
6. Enable **Notebook access**.

---

### Step 3 — Read the Secret in Python

Use:

```python
from google.colab import userdata

api_key = userdata.get("GEMINI_API_KEY")
```

Then create the Gemini client:

```python
from google import genai

client = genai.Client(
    api_key=api_key
)
```

Complete example:

```python
from google.colab import userdata
from google import genai

api_key = userdata.get("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)

print("Gemini API connected successfully.")
```

---

## ⚠️ API Key Security

Never publish code like this:

```python
api_key = "YOUR_REAL_API_KEY"
```

Do not commit your real API key to:

```text
README.md
.ipynb
.py
.txt
.env
```

Always use Colab Secrets or environment variables.

If a key is accidentally exposed, revoke it and create a new one.

---

## 📐 Cosine Similarity

The project calculates cosine similarity between the user query vector and every chunk vector.

The formula is:

```text
                         A · B
Cosine Similarity = ───────────────
                    ||A|| × ||B||
```

The implementation is:

```python
def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity
```

A higher score means the query and chunk are more semantically similar.

---

## 🔍 Semantic Search

The user enters a question:

```python
query = input(
    "Enter your question: "
)
```

The query is converted into an embedding using the same Gemini model:

```python
query_embedding = get_embedding(
    query
)
```

The program then compares the query embedding with every chunk embedding.

---

## 🏆 Top-3 Retrieval

The chunks are ranked from highest similarity to lowest.

```python
top_indices = np.argsort(
    similarities
)[::-1][:3]
```

Example output:

```text
============================================================
QUESTION
============================================================

What is the most popular Saudi food?

============================================================
TOP 3 MOST RELEVANT ANSWERS
============================================================

Rank: 1
Chunk Number: 4
Cosine Similarity Score: 0.8124

Answer:
Kabsa is one of the most famous dishes in Saudi Arabia...

------------------------------------------------------------

Rank: 2
Chunk Number: 5
Cosine Similarity Score: 0.7412

Answer:
Mandi is another popular rice and meat dish...

------------------------------------------------------------

Rank: 3
Chunk Number: 3
Cosine Similarity Score: 0.6891

Answer:
Jareesh is a traditional Saudi dish...
```

---

## 🚀 Getting Started

### 1. Open Google Colab

Create a new notebook or open the project notebook in Google Colab.

---

### 2. Install the Gemini SDK

Run:

```python
!pip install -q google-genai
```

---

### 3. Add Your Gemini API Key

Add this secret in Colab:

```text
GEMINI_API_KEY
```

Then enable notebook access.

---

### 4. Run the Notebook

The workflow is:

```text
Connect to Gemini
      ↓
Upload TXT File
      ↓
Read Document
      ↓
Create Chunks
      ↓
Generate Embeddings
      ↓
Enter User Question
      ↓
Generate Query Embedding
      ↓
Calculate Cosine Similarity
      ↓
Rank Chunks
      ↓
Return Top 3 Results
```

---

## 📂 Repository Structure

```text
gemini-semantic-search/
│
├── README.md
├── gemini_semantic_search.ipynb
├── sample_data/
│   └── sample.txt
└── requirements.txt
```

---

## 📦 requirements.txt

```text
google-genai
numpy
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 📓 Google Colab | Notebook environment |
| ✨ Gemini API | Embedding generation |
| 🔢 NumPy | Vector operations |
| 📐 Cosine Similarity | Semantic comparison |
| 📄 TXT | Input document |

---

## ✅ Project Requirements

| Requirement | Implementation |
|---|---|
| Load text document | ✅ |
| Fixed-size chunking | ✅ |
| Chunk overlap | ✅ 15% |
| Generate embeddings | ✅ Gemini |
| Convert text to vectors | ✅ |
| Accept user question | ✅ |
| Use same embedding model | ✅ |
| Calculate cosine similarity | ✅ |
| Rank chunks | ✅ |
| Return Top 3 chunks | ✅ |
| Display similarity scores | ✅ |
| Display Question + Answers | ✅ |

---

## 📊 Parameters

| Parameter | Value |
|---|---:|
| Chunk Size | 500 characters |
| Overlap | 15% |
| Overlap Size | 75 characters |
| Step Size | 425 characters |
| Top-K | 3 |
| Embedding Model | `gemini-embedding-001` |
| Task Type | `SEMANTIC_SIMILARITY` |

---

## ⚠️ Important Note

This project uses Gemini only for **embedding generation**.

It does not use Gemini to generate a new final answer.

Instead, it retrieves the three chunks that are most semantically similar to the user's question.

The project focuses on:

- text chunking
- vectorization
- embeddings
- cosine similarity
- semantic retrieval
- Top-K ranking

---

## 📌 Repository Information

### Repository Name

```text
gemini-semantic-search
```

### GitHub Description

```text
Semantic search pipeline using Gemini embeddings, overlapping text chunking, cosine similarity, and Top-K retrieval in Google Colab.
```

### Suggested Topics

```text
gemini
semantic-search
embeddings
cosine-similarity
information-retrieval
google-colab
python
rag
nlp
vector-search
```

---

<div align="center">

## 🔎 Gemini Semantic Search

Built with **Python • Gemini API • NumPy • Google Colab**

**Semantic retrieval using dense vector embeddings and cosine similarity.**

</div>
