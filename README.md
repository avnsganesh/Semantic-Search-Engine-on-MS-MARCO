**Semantic Search Engine on MS MARCO**

Comparing three retrieval approaches — Word2Vec with query expansion, SBERT (bi-encoder), and a Hybrid pipeline — on the MS MARCO v2.1 dataset.


Course project · IIT Gandhinagar · 2024–25




Team


Atragada Venkata Naga Sai Ganesh (24110060)
Manemoni Sumanjali (24110195)
Sidda Divya (24110337)
Divyam Sahu (25120028)



Overview

Standard keyword search fails when the query and document use different words for the same concept — the vocabulary mismatch problem. This project builds a semantic search engine that maps queries and passages into a shared vector space so that meaning-level similarity can be measured.

We implement and compare three designs on ~1.99M passages from MS MARCO v2.1. Design 1 uses Word2Vec with query expansion and FAISS indexing. Design 2 uses SBERT (all-MiniLM-L6-v2) to encode full sentences contextually. Design 3 is a Hybrid that uses Word2Vec for fast first-stage retrieval and SBERT for reranking.


Dataset

We used the MS MARCO v2.1 dataset from HuggingFace, taking the first 200K training samples which produced 1,995,277 passages. Of the 500 evaluation queries, 195 had no relevant passage marked (is_selected = 1) and were skipped — all metrics are reported over the remaining 305 queries.


Methodology

All three designs share the same preprocessing pipeline — lowercasing, punctuation removal, NLTK tokenization, and stopword filtering. Passages were processed in 4 batches of 500K and checkpointed to Google Drive to handle memory limits on Colab. All designs use FAISS IndexFlatIP with L2-normalized vectors for cosine similarity search.

pythondef preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS and t.isalpha()]

Design 1 — Word2Vec + Query Expansion

A skip-gram Word2Vec model was trained with Gensim on the full corpus (vector_size=300, window=5, min_count=2, 5 epochs), producing a vocabulary of 414K tokens. Each passage is represented as the average of its word vectors, stored as a float32 np.memmap file. At query time, each token is expanded with its top-N Word2Vec neighbours (cosine similarity ≥ 0.5); original tokens get weight 1.0 and expanded tokens get 0.7, keeping the query vector close to its original meaning.

Design 2 — SBERT Bi-Encoder

Uses all-MiniLM-L6-v2, a 6-layer transformer that produces 384-dim sentence embeddings. All 1.99M passages were encoded in batches of 512 and stored as a float32 np.memmap. At query time, the raw query is passed directly to SBERT and searched against the FAISS index — no preprocessing needed.

Design 3 — Hybrid

Combines the speed of Word2Vec with the accuracy of SBERT. Word2Vec first retrieves the top-300 candidates, then SBERT reranks only those 300 using a temporary per-query FAISS index. The trade-off: if a relevant passage doesn't appear in the Word2Vec top-300, SBERT never sees it. This is the hard recall ceiling of the approach.


Results

SBERT is the clear winner, achieving NDCG@5 of 0.438, Recall@5 of 0.606, and MRR@5 of 0.386 — roughly 6× better than Word2Vec on precision and recall. Word2Vec is the fastest at 933 ms since there's no neural model at query time, just a vector lookup and FAISS search. SBERT runs a full transformer forward pass per query, coming in at 1623 ms. The Hybrid sits in between at 1147 ms since it runs SBERT only on 300 candidates rather than 1.99M passages.

The Hybrid substantially improves over Word2Vec (MRR@5 goes from 0.047 to 0.226) but cannot match standalone SBERT because of the Stage 1 recall ceiling.

Query Expansion Ablation (Word2Vec)

We tested expansion sizes N ∈ {3, 5, 10}. N=3 performed best across all metrics (NDCG@5: 0.072, Recall@5: 0.106). Performance dropped consistently as N increased — at N=10, P@5 fell 44% compared to N=3. Larger N brings in loosely related words that dilute the query vector and pull in irrelevant passages.


Web App

Built an interactive Gradio demo where users can type any query, select a retrieval design, and adjust top-k. Each result shows the rank, cosine similarity score, full passage text, and query latency.


Tech Stack

gensim 4.4.0 · sentence-transformers 5.4.0 · faiss-cpu 1.13.2 · nltk 3.x · numpy 2.0.2 · datasets · gradio · Google Colab + Drive


Limitations

Word2Vec assigns one fixed vector per word, so it cannot handle polysemy, and its vocabulary is limited to what appeared during training. SBERT was not fine-tuned on MS MARCO specifically, so a domain-adapted model would likely do better; also, IndexFlatIP performs brute-force O(N) search and won't scale to very large corpora without approximate indexing (e.g., HNSW). Hybrid quality is entirely bounded by Word2Vec's Stage 1 recall. Evaluation excluded 195 queries with no ground truth, and latency was measured on a single Colab CPU run over a single query.


References


Johnson et al. (2017). FAISS – Facebook AI Similarity Search
Manning, Raghavan & Schütze (2008). Introduction to Information Retrieval. Cambridge University Press.
Microsoft. MS MARCO Dataset
Reimers & Gurevych (2019). Sentence Transformers
Řehůřek & Sojka (2010). Gensim Word2Vec
