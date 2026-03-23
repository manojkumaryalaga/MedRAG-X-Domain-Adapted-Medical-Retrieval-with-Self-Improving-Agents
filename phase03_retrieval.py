import numpy as np
import json
import math
import time
from collections import defaultdict
from sentence_transformers import SentenceTransformer


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1        = k1
        self.b         = b
        self.corpus    = []
        self.doc_freqs = []
        self.idf       = {}
        self.doc_len   = []
        self.avgdl     = 0
        self.vocab     = defaultdict(int)

    def fit(self, corpus):
        self.corpus = corpus
        N = len(corpus)
        for doc in corpus:
            tokens = doc.lower().split()
            self.doc_len.append(len(tokens))
            freq = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self.doc_freqs.append(freq)
            for t in set(tokens):
                self.vocab[t] += 1
        self.avgdl = sum(self.doc_len) / N
        for term, df in self.vocab.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        print(f"BM25 fitted | docs: {N} | vocab: {len(self.vocab)}")

    def score(self, query, doc_idx):
        tokens = query.lower().split()
        score  = 0.0
        dl     = self.doc_len[doc_idx]
        freq   = self.doc_freqs[doc_idx]
        for term in tokens:
            if term not in self.idf:
                continue
            tf  = freq.get(term, 0)
            num = self.idf[term] * tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += num / den
        return score

    def retrieve(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    def __init__(self, alpha=0.7):
        self.alpha      = alpha
        self.corpus     = []
        self.embeddings = None
        self.doc_freqs  = []
        self.idf        = {}
        self.doc_len    = []
        self.avgdl      = 0
        self.vocab      = defaultdict(int)
        self.k1         = 1.5
        self.b          = 0.75

    def fit(self, corpus, embeddings):
        self.corpus     = corpus
        self.embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-10)
        N = len(corpus)
        for doc in corpus:
            tokens = doc.lower().split()
            self.doc_len.append(len(tokens))
            freq = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self.doc_freqs.append(freq)
            for t in set(tokens):
                self.vocab[t] += 1
        self.avgdl = sum(self.doc_len) / N
        for term, df in self.vocab.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        print(f"HybridRetriever fitted | docs: {N} | vocab: {len(self.vocab)}")

    def _bm25_scores(self, query):
        tokens = query.lower().split()
        scores = np.zeros(len(self.corpus))
        for i in range(len(self.corpus)):
            dl   = self.doc_len[i]
            freq = self.doc_freqs[i]
            for term in tokens:
                if term not in self.idf:
                    continue
                tf  = freq.get(term, 0)
                num = self.idf[term] * tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += num / den
        max_s = scores.max()
        return scores / max_s if max_s > 0 else scores

    def _dense_scores(self, query_emb):
        query_emb = np.array(query_emb, dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        scores    = self.embeddings @ query_emb
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    def retrieve(self, query, query_emb, top_k=10):
        hybrid_s = (1 - self.alpha) * self._bm25_scores(query) + self.alpha * self._dense_scores(query_emb)
        top_idx  = np.argsort(hybrid_s)[::-1][:top_k]
        return [(hybrid_s[i], i) for i in top_idx]


def benchmark(retriever, triplets, model, alpha, label):
    retriever.alpha = alpha
    hits      = 0
    latencies = []
    for t in triplets:
        q_emb = model.encode([t['query']])[0]
        t0    = time.perf_counter()
        results = retriever.retrieve(t['query'], q_emb, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        for _, idx in results:
            if retriever.corpus[idx][:50] == t['positive'][:50]:
                hits += 1
                break
    recall  = hits / len(triplets)
    avg_lat = round(sum(latencies) / len(latencies), 1)
    print(f"{label:25s} | Recall@10: {recall:.3f} | Avg latency: {avg_lat}ms")
    return recall


def main():
    with open('triplets_5k.json') as f:
        triplets = json.load(f)
    corpus     = [t['positive'] for t in triplets]
    embeddings = np.load('embeddings.npy')

    print("Loading fine-tuned model...")
    model = SentenceTransformer("manojkumaryalaga/medrag-x-pubmedbert-v3")

    retriever = HybridRetriever(alpha=0.7)
    retriever.fit(corpus, embeddings)

    test_triplets = triplets[:50]
    print("\nBenchmark results:")
    print("-" * 65)
    benchmark(retriever, test_triplets, model, 0.0, "BM25 only")
    benchmark(retriever, test_triplets, model, 1.0, "Dense only")
    benchmark(retriever, test_triplets, model, 0.5, "Hybrid (alpha=0.5)")
    benchmark(retriever, test_triplets, model, 0.7, "Hybrid (alpha=0.7)")
    print("-" * 65)


if __name__ == "__main__":
    main()
