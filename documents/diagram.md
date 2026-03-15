                 ┌──────────────────────────────┐
                 │        USER QUESTION         │
                 └──────────────┬───────────────┘
                                ↓
                        ┌───────────────┐
                        │   RETRIEVAL   │   ⭐⭐⭐⭐⭐
                        │ (Vector Search)│
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │   CHUNKING    │   ⭐⭐⭐⭐⭐
                        │ (Size/Overlap)│
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │  EMBEDDINGS   │   ⭐⭐⭐⭐⭐
                        │  (Encoding)   │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │ VECTOR INDEX  │   ⭐⭐⭐⭐⭐
                        │ (Storage)     │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │ CONTEXT BUILD │   ⭐⭐⭐⭐
                        │ (Top-K Prompt)│
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │      LLM      │   ⭐⭐⭐⭐
                        │  Generation   │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │   ANSWER OUT  │   ⭐⭐⭐⭐
                        │ + Sources     │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │  EVALUATION   │   ⭐⭐⭐
                        │ (Quality)     │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │ OBSERVABILITY │   ⭐⭐⭐
                        │ Logs/Metrics  │
                        └───────┬───────┘
                                ↓
                        ┌───────────────┐
                        │ ITERATION     │   ⭐⭐⭐⭐⭐
                        │ (Tuning Loop) │
                        └───────────────┘


## 🧠 USER QUESTION PROCESSING

- Normalize text (lowercase, trim spaces, remove noise)
- Detect intent  
  - question  
  - summary  
  - comparison  
  - lookup  
- Reformulate query to be **knowledge-base friendly**
- Expand query with synonyms / domain keywords         

**Example**

User question:
> What does this document say about revenue?

Reformulated query:
> financial performance revenue growth earnings results

## 📚 DOCUMENT INGESTION / KNOWLEDGE BASE

- Create repo folder structure  
  - /data/raw  
  - /data/processed  
  - /models  
  - /logs
  - Load files (PDF, TXT, HTML, MD)
- Extract clean text  
- remove headers / footers  
- remove page numbers  
- remove navigation artifacts  
- Attach metadata  
- document_id  
- file_name  
- section  
- page  
- topic (if available)
- Store **raw + cleaned** versions

---

## CHUNKING STRATEGY

- Define chunk size (example: **500–1000 tokens**)
- Define overlap (**10–20%**)
- Preserve semantic boundaries  
- paragraph based  
- heading aware  
- Save chunk metadata  
- doc_id  
- chunk_id  
- position  
- Validate chunk quality  
- avoid very small chunks  
- avoid huge chunks  

---


## EMBEDDINGS

- Select embedding model
- Generate embeddings for each chunk
- Use batching for performance
- Store vectors with metadata
- Handle updates  
- re-embed changed documents  

## VECTOR INDEX

- Choose vector database  
- FAISS  
- Chroma  
- Weaviate  
- Create index structure
- Insert chunk embeddings
- Enable similarity search (**Top-K retrieval**)
- Persist index locally

---

## KEYWORD INDEX (Hybrid Search)

- Build **BM25 / sparse index**
- Index raw chunk text
- Enable keyword filtering
- Combine keyword + vector scores
- Tune hybrid weights

---

##  RETRIEVAL

- Accept reformulated query
- Generate query embedding
- Run vector similarity search
- Run keyword search
- Merge and rank results
- Return **Top-K relevant chunks**

---

## RE-RANKING (Phase-2 Improvement)

- Use cross-encoder or LLM scoring
- Improve relevance ordering
- Remove near-duplicate chunks
- Select final context set

---

## CONTEXT BUILDING (Prompt Assembly)

- Limit total token size
- Order chunks logically
- Add metadata citations
- Create system + user prompt template
- Inject retrieved context into prompt

---

## LLM GENERATION

- Send prompt + context to model
- Control temperature / max tokens
- Add grounding instruction  
> Answer only from provided context
- Add fallback behavior  
> If answer not found → say “Not found in knowledge base”

---

## 📤 ANSWER FORMATTING

- Return final answer
- Attach sources  
- document name  
- chunk id  
- Highlight confidence or relevance
- Provide fallback response if retrieval failed

---

## 📊 EVALUATION

- Create manual QA test set
- Measure retrieval relevance
- Measure answer correctness
- Track latency + token usage
- Maintain benchmark dataset

---

## 📡 OBSERVABILITY

- Log user queries
- Log retrieved chunks
- Log prompts and model responses
- Track failures / empty retrieval
- Monitor performance metrics
- Store experiment configurations

---

## 🔁 ITERATION LOOP (Most Important Engineering Habit)

- Tune chunk size / overlap
- Tune Top-K retrieval
- Tune hybrid scoring weights
- Improve prompt template
- Upgrade embedding / reranker models
- Add caching layer
- Repeat evaluation cycle

---