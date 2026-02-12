import pickle
from typing import List, Dict, Any
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import torch

from utils import save_to_pickle, tag_to_idx
from const import model_name


def load_inspec_dataset() -> Dict[str, Any]:
    """
    Load the MIDAS/Inspec dataset (extraction split)
    Returns:
        DatasetDict with train/validation/test splits
    """
    print("Loading MIDAS/Inspec dataset...")
    dataset = load_dataset("midas/inspec", "extraction")
    print(f"Dataset loaded. Splits available: {list(dataset.keys())}")
    return dataset


def bio_to_onehot(bio_tags: List[str]) -> np.ndarray:
    """
    Convert a list of BIO tags to one-hot encoded matrix

    Args:
        bio_tags: List of strings like ['O', 'B', 'I', 'O', ...]

    Returns:
        numpy array of shape (sequence_length, 3)
        columns: [O, B, I]
    """
    # We'll use simple 3-class one-hot: O / B / I
    # (ignoring the specific entity type for this conversion)
    onehot = np.zeros((len(bio_tags), 3), dtype=np.int32)

    for i, tag in enumerate(bio_tags):
        onehot[i, tag_to_idx[tag]] = 1

    return onehot


def convert_example_to_processed(example: Dict, tokenizer: Any, device: Any) -> Dict:
    """
    Process one example from the dataset:
    - keep id and document (tokens)
    - convert doc_bio_tags to one-hot

    Returns:
        Dictionary with: id, tokens, labels_onehot
    """
    tokenized_document = tokenizer(
        example["document"], is_split_into_words=True, return_tensors="pt"
    )
    bio_tags = example["doc_bio_tags"]

    labels_onehot = bio_to_onehot(bio_tags)

    return {
        "id": example["id"],
        "tokens": tokenized_document["input_ids"],
        "attention_mask": tokenized_document["attention_mask"],
        "labels_onehot": labels_onehot,
    }


def process_split(dataset_split: Any, tokenizer: Any, device: Any) -> List[Dict]:
    """
    Process an entire dataset split (train / validation / test)
    """
    print(f"Processing split with {len(dataset_split)} examples...")

    processed_examples = []

    for i, example in enumerate(dataset_split):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i} examples...")
        processed = convert_example_to_processed(example, tokenizer, device)
        processed_examples.append(processed)

    print(f"Finished processing split. Got {len(processed_examples)} examples.")
    return processed_examples


def main():
    # 1. Load dataset
    raw_dataset = load_inspec_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 2. Process each split
    processed_dataset = {}

    for split_name in ["train", "validation", "test"]:
        if split_name in raw_dataset:
            processed_dataset[split_name] = process_split(
                raw_dataset[split_name], tokenizer, device
            )

    # 3. Save to pickle
    save_to_pickle(processed_dataset, "inspec_onehot_encoded.pkl")

    # Optional: quick summary
    for split, examples in processed_dataset.items():
        print(f"{split}: {len(examples)} examples")
        if examples:
            print(
                f"  Example shape of first labels_onehot: {examples[0]['labels_onehot'].shape}"
            )


if __name__ == "__main__":
    main()
