import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class CustomTokenClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_prob: float = 0.1):
        super(CustomTokenClassifier, self).__init__()
        # Load the base transformer model (without the classification head)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = self.base_model.config.hidden_size
        # Define your own classification head: a linear layer mapping hidden states to num_labels
        self.fc1 = nn.Linear(hidden_size, num_labels)
        # Optionally, define a softmax layer (note: during training you usually pass logits to CrossEntropyLoss)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the last hidden states (shape: [batch_size, seq_length, hidden_size])
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.fc1(sequence_output)  # [batch_size, seq_length, num_labels]
        
        # Optionally, compute probabilities using softmax
        return self.softmax(logits)        
    
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.base_model.named_parameters():
            param[1].requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.base_model.named_parameters():
            param[1].requires_grad=True

def bio_to_ohe(bio):
    if bio == "B":
        return [1, 0, 0]
    elif bio == "I":
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def fix_labels(tokenized_inputs, bio_labels, debug=False):
    word_ids = tokenized_inputs.word_ids()

    new_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # Special token (e.g. [CLS], [SEP]) get a dummy label (commonly -100 so they're ignored in loss)
            new_labels.append(-100)
        elif word_idx != prev_word_idx:
            # First token of a new word: assign its original BIO tag.
            new_labels.append(bio_labels[word_idx])
        else:
            # If the label starts with "B", change it to "I":
            label = bio_labels[word_idx]
            if label.startswith("B"):
                label = label.replace("B", "I")
            new_labels.append(label)
        prev_word_idx = word_idx

    if debug:
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids.reshape(tokenized_inputs.input_ids.shape[1]))
        for token, label in zip(tokens, new_labels):
            print(f"{token:10s} -> {label}")

    new_labels = [[bio_to_ohe(item) for item in new_labels]]
    new_labels = torch.tensor(new_labels)

    return new_labels

def training_step(dataset, model, optimizer, loss_fn, tokenizer, device):
    epoch_loss = 0
    for i, item in enumerate(dataset):
        text = item["document"].to(device)
        bio_labels = item["doc_bio_tags"].to(device)

        labels = fix_labels(tokenized_inputs, bio_labels)
        
        # Forward pass (in training mode, labels are provided so the loss is computed)
        output = model(tokenized_inputs.input_ids, tokenized_inputs.attention_mask)

        output = output.permute(0, 2, 1).type(torch.float32)
        labels = labels.permute(0, 2, 1).type(torch.float32)

        optimizer.zero_grad()

        loss = loss_fn(output, labels)
        epoch_loss += loss
        
        loss.backward()
        optimizer.step()
    return epoch_loss

class TweetDataset(torch.utils.data.Dataset):
    """
    Class to store the tweet data as PyTorch Dataset
    """
    
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels
        
    def __getitem__(self, idx):
        # an encoding can have keys such as input_ids and attention_mask
        # item is a dictionary which has the same keys as the encoding has
        # and the values are the idxth value of the corresponding key (in PyTorch's tensor format)
        item = {} #{key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item["document"] = torch.tensor(self.encodings[idx])
        item["attention_mask"] = torch.tensor(self.attention_masks[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Example usage:
if __name__ == "__main__":
    model_name = "answerdotai/ModernBERT-base"  # or any other pretrained model
    num_labels = 3  # For example, if you have tags like B, I, and O
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used: {}.".format(device))

    model = CustomTokenClassifier(model_name, num_labels)
    model.freeze_bert()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=27e-6)

    dataset = load_dataset("midas/inspec", "extraction")
    
    loss_fn = nn.BCELoss()


    tokenized_inputs = [tokenizer(dataset["train"][j]["document"], 
                        is_split_into_words=True,
                        return_tensors="pt"
                        ) for j in range(len(dataset["train"]))]
    
    input_ids = [ti["input_ids"] for ti in tokenized_inputs]
    attention_masks = [ti["attention_mask"] for ti in tokenized_inputs]

    labels = []

    for i, item in enumerate(dataset["train"]):
        bio_labels = item["doc_bio_tags"]
        label = fix_labels(tokenized_inputs[i], bio_labels)
        labels.append(label)

    # train_loader = DataLoader(dataset["train"], batch_size=10, shuffle=True)
    # tensorDataset = TensorDataset(input_ids, attention_mask, labels)
    tensorDataset = TweetDataset(input_ids, attention_masks, labels)
    train_loader = DataLoader(tensorDataset, batch_size=24, shuffle=True)

    for epoch in tqdm(range(10)):
        loss = training_step(train_loader, model, optimizer, loss_fn, tokenizer, device)
        print(loss)
