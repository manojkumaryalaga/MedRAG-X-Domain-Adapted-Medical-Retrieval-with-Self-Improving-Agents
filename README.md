# MedRAG-X

> Domain-adapted medical retrieval system — fine-tuned PubMedBERT bi-encoder on PubMedQA, custom BM25 and dense retrieval built from scratch in pure numpy, and a self-improving multi-agent critic loop that automatically rewrites failed queries.

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=flat-square&logo=huggingface)
![WandB](https://img.shields.io/badge/W%26B-Training_Logs-FFBE00?style=flat-square&logo=weightsandbiases)
![Kaggle](https://img.shields.io/badge/Kaggle-T4_GPU-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Model on HuggingFace

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("manojkumaryalaga/medrag-x-pubmedbert-v3")
```

Live at: [huggingface.co/manojkumaryalaga/medrag-x-pubmedbert-v3](https://huggingface.co/manojkumaryalaga/medrag-x-pubmedbert-v3)

W&B training run: [wandb.ai/manojkyalaga-florida-atlantic-university/medrag-x](https://wandb.ai/manojkyalaga-florida-atlantic-university/medrag-x/runs/zdk0w6r7)

---

## Results

### Data pipeline output
![Data Pipeline](screenshots/data_pipeline.png)

### Training — epoch scores saved every checkpoint
![Training](screenshots/training.png)

### W&B training curves
![WandB](screenshots/wandb.png)

### Retrieval benchmark results
![Benchmark](screenshots/benchmark.png)

### Multi-agent self-improving loop
![Agents](screenshots/agents.png)

### HuggingFace model page
![HuggingFace](screenshots/huggingface.png)

---

## Benchmark Results

### Training convergence — 5,000 samples, strict train/eval split

| Epoch | NDCG@10 | Checkpoint |
|---|---|---|
| 1 | 0.9708 | saved |
| 2 | 0.9682 | saved |
| 3 | 0.9672 | saved |
| 4 | **0.9743** | **best — used for all benchmarks** |
| 5 | 0.9693 | saved |

Training: 4,000 samples | Eval: 1,000 samples (no overlap) | Dataset: 62,249 PubMedQA pairs available

### Retrieval benchmark — 50 PubMedQA queries

| System | Recall@10 ↑ | Avg Latency ↓ |
|---|---|---|
| **MedRAG-X Hybrid (alpha=0.7)** | **1.000** | **63.8ms** |
| Dense only (fine-tuned) | 1.000 | 65.3ms |
| BM25 only (from scratch) | 0.940 | 63.7ms |

### HuggingFace eval metrics (1,000 sample eval set)

| Metric | Value |
|---|---|
| Cosine NDCG@10 | 0.968 |
| Cosine Accuracy@1 | 0.945 |
| Cosine Accuracy@10 | 0.989 |
| Cosine Recall@10 | 0.989 |
| Cosine MRR@10 | 0.962 |

### Multi-agent self-improvement

| Query | Round 1 | Round 2 | Result |
|---|---|---|---|
| Easy — clear medical terms | 8/10 | — | accepted round 1 |
| Medium — ambiguous phrasing | 7/10 | — | accepted round 1 |
| Hard — rare terminology | 2/10 | 2/10 | rewritten twice |

> Note: Recall@10 saturates at 1.000 on this 50-query test set.
> BM25 dropping to 0.940 while dense holds 1.000 is the meaningful signal —
> the fine-tuned model handles semantic medical queries that keyword search misses.

---

## Architecture

```
PubMedQA Dataset (62,249 available — 5,000 used)
        │
        ▼
Hard Negative Mining
(query, positive, hard negative from different question)
4,000 train / 1,000 eval — strict no-overlap split
        │
        ▼
Fine-tuning — PubMedBERT bi-encoder
MultipleNegativesRankingLoss | batch_size=32 | 5 epochs
Best checkpoint: Epoch 4 | NDCG@10 = 0.9743
        │
        ▼
Custom Retrieval Engine (pure numpy — no FAISS, no ChromaDB)
├── BM25 from scratch    (k1=1.5, b=0.75, IDF weighting)
├── Dense cosine search  (normalized embeddings, matrix multiply)
└── Hybrid fusion        (alpha=0.7 dense + 0.3 BM25)
Recall@10 = 1.000 | Latency = 63.8ms
        │
        ▼
Multi-Agent Self-Improving Loop
├── RetrieverAgent  — hybrid retrieval
├── CriticAgent     — LLM relevance scoring (0-10)
└── RewriterAgent   — automatic query rewriting on score < 7
        │
        ▼
Model pushed to HuggingFace Hub (15 downloads)
Training logged to Weights & Biases
```

---

## Tech Stack

### Model Training
![PubMedBERT](https://img.shields.io/badge/PubMedBERT-Base-blue?style=flat-square)
![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-5.x-orange?style=flat-square)
![WandB](https://img.shields.io/badge/W%26B-Experiment_Tracking-yellow?style=flat-square)

### Retrieval (from scratch)
![NumPy](https://img.shields.io/badge/NumPy-Pure_Math-013243?style=flat-square&logo=numpy)
![BM25](https://img.shields.io/badge/BM25-Custom_Implementation-red?style=flat-square)

### Agents
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=flat-square&logo=openai)

### Data & Infrastructure
![HuggingFace](https://img.shields.io/badge/PubMedQA-62k_samples-FFD21E?style=flat-square&logo=huggingface)
![Kaggle](https://img.shields.io/badge/Kaggle-T4_GPU-20BEFF?style=flat-square&logo=kaggle)

---

## Project Structure

```
MedRAG-X/
├── screenshots/
│   ├── data_pipeline.png
│   ├── training.png
│   ├── wandb.png
│   ├── benchmark.png
│   ├── agents.png
│   └── huggingface.png
├── phase01_data_pipeline.py
├── phase02_training.py
├── phase03_retrieval.py
├── phase04_agents.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/manojkumaryalaga/MedRAG-X
cd MedRAG-X
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API keys

```
OPENAI_API_KEY=sk-your-key-here
WANDB_API_KEY=your-wandb-key-here
```

### 3. Run data pipeline

```bash
python phase01_data_pipeline.py
```

```
Loading PubMedQA...
Total available: 62249
Built 5000 triplets
Train : 4000 | Eval : 1000
```

### 4. Train model

```bash
python phase02_training.py
```

```
Epoch 1 | NDCG@10: 0.9708
Epoch 2 | NDCG@10: 0.9682
Epoch 3 | NDCG@10: 0.9672
Epoch 4 | NDCG@10: 0.9743  <- best
Epoch 5 | NDCG@10: 0.9693
Best NDCG@10: 0.9743
```

### 5. Run retrieval benchmark

```bash
python phase03_retrieval.py
```

```
BM25 only                  | Recall@10: 0.940 | Avg latency: 63.7ms
Dense only                 | Recall@10: 1.000 | Avg latency: 65.3ms
Hybrid (alpha=0.7)         | Recall@10: 1.000 | Avg latency: 63.8ms
```

### 6. Run multi-agent loop

```bash
python phase04_agents.py
```

```
Round 1 | Critic: 8/10 | Accepted at round 1
Round 1 | Critic: 2/10 | Rewrite needed
Rewritten: What are the causes of syncope in infants...
Round 2 | Critic: 2/10 | Accepted
```

---

## Key Design Decisions

**Why MultipleNegativesRankingLoss over TripletLoss?**
MNRL uses all other samples in the batch as negatives. With batch_size=32 you get 31 negatives per sample instead of 1 — much stronger training signal. First run with TripletLoss failed — this is what fixed it.

**Why build BM25 from scratch?**
Shows understanding of IDF weighting, document length normalization, and k1/b hyperparameters — the exact knowledge FAANG system design interviews test.

**Why alpha=0.7 for hybrid retrieval?**
Empirically 0.7 dense + 0.3 BM25 gives best Recall@10. Dense captures semantic medical terminology; BM25 handles exact term matching.

**Why does NDCG vary across epochs?**
Small eval set variance — 1,000 samples gives ~±0.01 standard error. Best checkpoint at epoch 4 is saved and used for all benchmarks.

---

## Limitations

- Eval set of 1,000 samples has ~±0.01 standard error
- Hard negatives are randomly sampled, not semantically mined
- Retrieval benchmark uses same PubMedQA distribution as training
- Multi-agent loop tested on 5 queries — larger scale evaluation needed

---

## What I Learned

- MultipleNegativesRankingLoss outperforms TripletLoss significantly on small datasets
- BM25 from scratch achieves 0.940 Recall@10 with zero training — extremely strong baseline
- NDCG varies across epochs on small eval sets — always save best checkpoint not last
- Self-improving agent loops need careful stopping criteria
- Eval set design is as important as model design

---

## Hashtags

`#MedicalAI` `#NLP` `#RAG` `#LLM` `#PubMed` `#BioNLP` `#HuggingFace` `#SentenceTransformers` `#PubMedBERT` `#MultiAgentAI` `#InformationRetrieval` `#BM25` `#VectorSearch` `#Python` `#MachineLearning` `#DeepLearning` `#AIEngineering` `#DataScience` `#HealthcareAI` `#ClinicalNLP` `#PortfolioProject` `#BuildInPublic` `#OpenSource` `#GitHub` `#WandB` `#FineTuning` `#AIResearch` `#RetrievalAugmentedGeneration` `#FAANG` `#DataScientist`

---

## Author

**Manoj Kumar Yalaga** — AI/Data Scientist

[![HuggingFace](https://img.shields.io/badge/HuggingFace-manojkumaryalaga-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/manojkumaryalaga)
[![GitHub](https://img.shields.io/badge/GitHub-manojkumaryalaga-181717?style=flat-square&logo=github)](https://github.com/manojkumaryalaga)
[![WandB](https://img.shields.io/badge/W%26B-Training_Logs-FFBE00?style=flat-square)](https://wandb.ai/manojkyalaga-florida-atlantic-university/medrag-x)

---

*Built with PubMedBERT · NumPy · SentenceTransformers · OpenAI · Weights & Biases*
