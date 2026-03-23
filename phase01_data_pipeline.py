from datasets import load_dataset
import random
import json
from pathlib import Path


def load_pubmedqa():
    print("Loading PubMedQA...")
    labeled   = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    all_data  = list(labeled['train']) + list(unlabeled['train'])
    print(f"Total available: {len(all_data)}")
    return all_data


def build_triplets(data, n=5000):
    print(f"Building {n} triplets...")
    triplets = []
    total    = len(data)

    for i, sample in enumerate(data):
        if len(triplets) >= n:
            break
        question = sample['question']
        contexts = sample['context']['contexts']
        if not contexts:
            continue
        positive      = " ".join(contexts[:2])[:512]
        neg_idx       = random.choice([j for j in range(min(total, 3000)) if j != i])
        neg_contexts  = data[neg_idx]['context']['contexts']
        if not neg_contexts:
            continue
        hard_negative = " ".join(neg_contexts[:2])[:512]
        triplets.append({
            "query":    question,
            "positive": positive,
            "negative": hard_negative,
        })

    print(f"Built {len(triplets)} triplets")
    return triplets


def main():
    Path("./data").mkdir(exist_ok=True)
    data     = load_pubmedqa()
    triplets = build_triplets(data, n=5000)
    train    = triplets[:4000]
    eval_    = triplets[4000:]

    with open('/kaggle/working/triplets_5k.json', 'w') as f:
        json.dump(triplets, f)

    print(f"Saved triplets_5k.json")
    print(f"Train : {len(train)}")
    print(f"Eval  : {len(eval_)}")


if __name__ == "__main__":
    main()
