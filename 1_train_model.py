import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter

# Assuming these are defined in your files
from model import BERT_CRF, START_TAG

MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 24
LR = 1e-6
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG_TO_IDX = {"B": 0, "I": 1, "O": 2, START_TAG: 3}
IX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}

def get_word_level_bio_labels(example) -> List[int]:
    """Convert string BIO tags → integer labels"""
    mapping = {"B": 0, "I": 1, "O": 2}
    return [mapping[tag] for tag in example["doc_bio_tags"]]


def align_labels_with_tokens(
    word_ids: List[int | None],
    word_level_labels: List[int],
) -> List[int]:
    """Standard word-piece label alignment: -100 on special tokens & subwords"""
    aligned = []
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(-100)
        elif word_idx != prev_word_idx:
            aligned.append(word_level_labels[word_idx])
        else:
            # subword → copy previous label (most common choice for NER)
            # You could also do: if original was B → make it I
            aligned.append(word_level_labels[word_idx])
        prev_word_idx = word_idx

    return aligned


class InspecDataset(Dataset):
    def __init__(self, encodings: List[Dict], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[idx]["input_ids"][0]
        item["attention_mask"] = self.encodings[idx]["attention_mask"][0]
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def evaluate_f1(
    model: BERT_CRF,
    dataloader: DataLoader,
    dataset,  # original huggingface dataset
    tokenizer,
    device,
) -> float:
    model.eval()
    TP = Counter()
    FP = Counter()
    FN = Counter()

    for batch, example in tqdm(zip(dataloader, dataset), desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # CRF decode
        _, predicted_tags = model(input_ids, attention_mask)  # assume returns best path

        # We compare at token level (including subwords)
        for pred_seq, true_seq, mask in zip(
            predicted_tags, batch["labels"], attention_mask
        ):
            for p, t, m in zip(pred_seq, true_seq, mask):
                if m == 0:
                    continue
                if t == -100:
                    continue
                p, t = p.item(), t.item()
                if p == t:
                    TP[t] += 1
                else:
                    FP[p] += 1
                    FN[t] += 1

    if not TP:
        return 0.0

    f1_scores = []
    for label in [0, 1, 2]:  # B, I, O
        p = TP[label] / (TP[label] + FP[label]) if TP[label] + FP[label] > 0 else 0
        r = TP[label] / (TP[label] + FN[label]) if TP[label] + FN[label] > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Macro F1: {macro_f1:.4f}")
    for i, tag in enumerate(["B", "I", "O"]):
        print(f"  {tag:2s}  P: {p:.3f}  R: {r:.3f}  F1: {f1_scores[i]:.3f}")

    return macro_f1


def main():
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Model
    model = BERT_CRF(
        model_name=MODEL_NAME,
        tag_to_idx=TAG_TO_IDX,
    )
    model.to(DEVICE)

    if os.path.exists("model_weights.pth"):
        model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
        print("Loaded existing weights")
    else:
        print("Starting from scratch (model_weights.pth not found)")

    model.freeze_bert()  # assuming this method exists
    optimizer = AdamW(model.parameters(), lr=LR)

    # Training Data
    ds = load_dataset("midas/inspec", "extraction")["train"]

    encodings = []
    
    for document in ds:
        encodings.append(tokenizer(
            document["document"],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ))

    aligned_labels = []
    for i in range(len(ds)):
        word_ids = encodings[i].word_ids()[:-1] # Removing EOS
        word_labels = get_word_level_bio_labels(ds[i])
        aligned = align_labels_with_tokens(word_ids, word_labels)
        aligned_labels.append(aligned)

    dataset = InspecDataset(encodings, aligned_labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Training
    best_f1 = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            loss = model.neg_log_likelihood(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d} | loss = {avg_loss:.4f}")

        # Evaluate every 10 epochs (or more/less often)
        if epoch % 10 == 0:
            current_f1 = evaluate_f1(model, loader, ds, tokenizer, DEVICE)
            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(model.state_dict(), "model_weights_best.pth")
                print(f"↑ New best macro-F1: {best_f1:.4f}")

        # Always save latest
        torch.save(model.state_dict(), "model_weights.pth")

    print("Training finished.")

    # Test
    ds = load_dataset("midas/inspec", "extraction")["test"]

    encodings = tokenizer(
        ds["document"],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )

    aligned_labels = []
    for i in range(len(ds)):
        word_ids = encodings.word_ids(batch_index=i)
        word_labels = get_word_level_bio_labels(ds[i])
        aligned = align_labels_with_tokens(word_ids, word_labels)
        aligned_labels.append(aligned)

    dataset = InspecDataset(encodings, aligned_labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_f1 = evaluate_f1(model, loader, ds, tokenizer, DEVICE)
    print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
