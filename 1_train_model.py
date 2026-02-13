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
BATCH_SIZE = 1
LR = 1e-6
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG_TO_IDX = {"B": 0, "I": 1, "O": 2, START_TAG: 3}
IX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}


class InspecDataset(Dataset):
    def __init__(self, encodings: List[Dict], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[idx]["input_ids"]
        item["attention_mask"] = self.encodings[idx]["attention_mask"]
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


def get_word_level_bio_labels(bio_tags) -> List[int]:
    """Convert string BIO tags → integer labels"""
    return [TAG_TO_IDX[tag] for tag in bio_tags["doc_bio_tags"]]


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


def get_encodings_and_aligned_labels(ds, tokenizer):
    """
    When a string is tokenized, there's no 1:1 mapping between a word and a token.
    Sometimes a word is being tokenized with more than 1 token.
    This is why we have to align the BIO labels to the encodings.
    """
    encodings = []
    aligned_labels = []

    for document in ds:
        tokenized_input = tokenizer(
            document["document"],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        encodings.append(
            {
                "input_ids": tokenized_input["input_ids"][0][:-1], # Get rid of [SEP]
                "attention_mask": tokenized_input["attention_mask"][0][:-1],
            }
        )

        word_ids = (
            tokenized_input.word_ids()
        )  # This array tells us which input_id belongs to which word
        word_labels = get_word_level_bio_labels(document)
        aligned = align_labels_with_tokens(word_ids, word_labels)[:-1]
        aligned[0] = TAG_TO_IDX[START_TAG] # Replace [CLS] with our custom value for it
        aligned_labels.append(aligned)

    return encodings, aligned_labels


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

    encodings, aligned_labels = get_encodings_and_aligned_labels(ds, tokenizer)
    # sanity check: same number of examples
    assert len(encodings) == len(
        aligned_labels
    ), f"Number of encodings ({len(encodings)}) != number of label lists ({len(aligned_labels)})"

    # find first mismatch (if any) and raise with informative message
    for i, (enc, lbl) in enumerate(zip(encodings, aligned_labels)):
        if len(enc["input_ids"]) != len(lbl):
            raise AssertionError(
                f"Length mismatch at index {i}: encodings length={len(enc)} vs labels length={len(lbl)}"
            )

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
