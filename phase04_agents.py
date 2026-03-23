import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from phase03_retrieval import HybridRetriever

load_dotenv()


class RetrieverAgent:
    def __init__(self, retriever, model, corpus):
        self.retriever = retriever
        self.model     = model
        self.corpus    = corpus

    def run(self, query, top_k=5):
        q_emb   = self.model.encode([query])[0]
        results = self.retriever.retrieve(query, q_emb, top_k=top_k)
        docs    = [self.corpus[idx] for _, idx in results]
        scores  = [round(float(s), 3) for s, _ in results]
        return docs, scores


class CriticAgent:
    def __init__(self, client):
        self.client = client

    def run(self, query, docs):
        preview = "\n".join([f"Doc {i+1}: {d[:200]}" for i, d in enumerate(docs)])
        prompt  = f"""Medical retrieval critic.
Query: {query}
Documents:
{preview}
Respond JSON only:
{{"doc1_relevant":true/false,"relevance_score":0-10,"rewrite_needed":true/false,"rewritten_query":"..."}}"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        try:
            text = response.choices[0].message.content.strip().replace("```json","").replace("```","")
            return json.loads(text)
        except Exception:
            return {"doc1_relevant": False, "relevance_score": 0, "rewrite_needed": True, "rewritten_query": query}


class RewriterAgent:
    def __init__(self, client):
        self.client = client

    def run(self, original_query, critic_feedback):
        if not critic_feedback.get("rewrite_needed"):
            return original_query
        rw = critic_feedback.get("rewritten_query", "")
        if rw and rw != original_query:
            return rw
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Rewrite this medical query to be more specific:\n{original_query}\nRewritten:"}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


class MedRAGXAgent:
    def __init__(self, ret, crit, rew):
        self.ret  = ret
        self.crit = crit
        self.rew  = rew
        self.log  = []

    def run(self, query, max_rounds=2):
        q = query
        for rnd in range(max_rounds):
            print(f"\n  Round {rnd+1} | {q[:80]}")
            docs, scores = self.ret.run(q)
            fb = self.crit.run(q, docs)
            print(f"  Critic: {fb.get('relevance_score')}/10 | Relevant: {fb.get('doc1_relevant')} | Rewrite: {fb.get('rewrite_needed')}")
            self.log.append({"round": rnd+1, "query": q, "feedback": fb})
            if fb.get("relevance_score", 0) >= 7 or not fb.get("rewrite_needed"):
                print(f"  Accepted at round {rnd+1}")
                break
            q = self.rew.run(q, fb)
            print(f"  Rewritten: {q[:80]}")
        return docs, self.log


def main():
    with open('triplets_5k.json') as f:
        triplets = json.load(f)
    corpus     = [t['positive'] for t in triplets]
    embeddings = np.load('embeddings.npy')

    model     = SentenceTransformer("manojkumaryalaga/medrag-x-pubmedbert-v3")
    retriever = HybridRetriever(alpha=0.7)
    retriever.fit(corpus, embeddings)

    client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ret_agent = RetrieverAgent(retriever, model, corpus)
    crit_agent= CriticAgent(client)
    rew_agent = RewriterAgent(client)
    agent     = MedRAGXAgent(ret_agent, crit_agent, rew_agent)

    test_queries = [t['query'] for t in triplets[:5]]

    print("=" * 60)
    print("MedRAG-X Multi-Agent Self-Improving Loop")
    print("=" * 60)

    for q in test_queries:
        print(f"\nQuery: {q[:80]}")
        docs, log = agent.run(q, max_rounds=2)
        print(f"Top doc: {docs[0][:100]}...")
        agent.log = []

    print("\nAll phases complete!")


if __name__ == "__main__":
    main()
