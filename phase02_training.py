import json
import wandb
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader


def load_and_prepare(path='triplets_5k.json'):
    with open(path) as f:
        triplets = json.load(f)
    train_triplets = triplets[:4000]
    eval_triplets  = triplets[4000:]
    examples = []
    for i, t in enumerate(train_triplets):
        examples.append(InputExample(texts=[t['query'], t['positive']]))
        neg1 = train_triplets[(i + 1) % len(train_triplets)]['positive']
        examples.append(InputExample(texts=[t['query'], t['positive']]))
        examples.append(InputExample(texts=[neg1, t['positive']]))
    return examples, train_triplets, eval_triplets


def build_evaluator(eval_triplets):
    queries  = {str(i): t['query']    for i, t in enumerate(eval_triplets)}
    corpus   = {str(i): t['positive'] for i, t in enumerate(eval_triplets)}
    relevant = {str(i): {str(i)}      for i in range(len(eval_triplets))}
    return evaluation.InformationRetrievalEvaluator(
        queries, corpus, relevant, name="pubmed-eval-1k"
    )


def train(
    model_name   = "NeuML/pubmedbert-base-embeddings",
    output_path  = "/kaggle/working/medrag-x-v3",
    batch_size   = 32,
    epochs       = 5,
    warmup_steps = 300,
):
    examples, train_triplets, eval_triplets = load_and_prepare()
    print(f"Training examples : {len(examples)}")
    print(f"Eval samples      : {len(eval_triplets)}")

    wandb.init(
        project="medrag-x",
        name="pubmedbert-v3-5k",
        config={
            "model":         model_name,
            "train_samples": len(examples),
            "eval_samples":  len(eval_triplets),
            "epochs":        epochs,
            "batch_size":    batch_size,
            "loss":          "MultipleNegativesRankingLoss",
        }
    )

    print(f"Loading {model_name}...")
    model      = SentenceTransformer(model_name)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn    = losses.MultipleNegativesRankingLoss(model=model)
    evaluator  = build_evaluator(eval_triplets)
    epoch_scores = []

    def save_and_log(score, epoch, steps):
        epoch_scores.append({"epoch": epoch, "ndcg": round(score, 4)})
        wandb.log({"epoch": epoch, "ndcg_score": score})
        model.save(f"/kaggle/working/ckpt_epoch_{epoch}")
        print(f"\nEpoch {epoch} | NDCG@10: {score:.4f} | Checkpoint saved")

    print(f"\nTraining {epochs} epochs...")
    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_path,
        callback=save_and_log,
    )
    wandb.finish()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"{'Epoch':<8} {'NDCG@10':<10}")
    print("-" * 20)
    for e in epoch_scores:
        marker = " <- best" if e['ndcg'] == max(x['ndcg'] for x in epoch_scores) else ""
        print(f"{e['epoch']:<8} {e['ndcg']:<10}{marker}")
    print(f"\nBest NDCG@10 : {max(e['ndcg'] for e in epoch_scores):.4f}")
    print(f"Model saved  : {output_path}")
    return model


if __name__ == "__main__":
    train()
