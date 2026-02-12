import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from tabulate import tabulate
import os

from model import BiLSTM_CRF, START_TAG, STOP_TAG
from dataset import MyDataset, collate
from utils import fix_labels, training_step, print_f_score

def main():
    # Model and tokenizer configuration.
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used: {}.".format(device))

    EMBEDDING_DIM = 5
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(model_name=model_name,
                       tag_to_ix=tag_to_ix, 
                       embedding_dim=EMBEDDING_DIM,
                       device=device)

    if os.path.exists('model_weights.pth'):
        model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    else:
        print("model_weights.pth not found. Please ensure the file exists before loading.")

    model.freeze_bert()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-6)

    # Load and tokenize dataset.
    dataset = load_dataset("midas/inspec", "extraction")
    tokenized_inputs = [
        tokenizer(dataset["train"][j]["document"],
                  is_split_into_words=True,
                  return_tensors="pt")
        for j in range(len(dataset["train"]))
    ]
    
    input_ids = [ti["input_ids"] for ti in tokenized_inputs]
    attention_masks = [ti["attention_mask"] for ti in tokenized_inputs]
    labels = []
    
    for i, item in enumerate(dataset["train"]):
        bio_labels = item["doc_bio_tags"]
        label = fix_labels(tokenized_inputs[i], bio_labels, tokenizer)
        labels.append(label)
    
    tensorDataset = MyDataset(input_ids, attention_masks, labels)
    train_loader = DataLoader(tensorDataset, batch_size=24, shuffle=True, collate_fn=collate)

    # Training loop.
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        if epoch % 10 == 0:
            print_f_score(dataset["train"], tokenizer, device, model)

        loss = training_step(train_loader, model, optimizer, device)
        print(f"Epoch {epoch} Loss: {loss.item()}")
        torch.save(model.state_dict(), 'model_weights.pth')
        
if __name__ == "__main__":
    main()
