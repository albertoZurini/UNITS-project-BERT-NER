import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

class CustomTokenClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_prob: float = 0.1):
        super(CustomTokenClassifier, self).__init__()
        # Load the base transformer model (without the classification head)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = self.base_model.config.hidden_size
        # Define your own classification head: a linear layer mapping hidden states to num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Optionally, define a softmax layer (note: during training you usually pass logits to CrossEntropyLoss)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states from the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the last hidden states (shape: [batch_size, seq_length, hidden_size])
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]
        
        # Optionally, compute probabilities using softmax
        probs = self.softmax(logits)
        
        if labels is not None:
            # Flatten the tokens for loss computation. Here, we assume that labels is already
            # aligned with the input tokens (and that padded tokens have a label value of -100).
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape logits to (batch_size * seq_length, num_labels)
            # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1, labels.size(-1)))

            logits = logits.permute(0, 2, 1).type(torch.float32)
            labels = labels.permute(0, 2, 1).type(torch.float32)

            loss = loss_fct(logits, labels)
            return loss, logits, probs
        
        return logits, probs

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
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids)
        for token, label in zip(tokens, new_labels):
            print(f"{token:10s} -> {label}")

    new_labels = [[bio_to_ohe(item) for item in new_labels]]
    new_labels = torch.tensor(new_labels)

    return new_labels



# Example usage:
if __name__ == "__main__":
    model_name = "answerdotai/ModernBERT-base"  # or any other pretrained model
    num_labels = 3  # For example, if you have tags like B, I, and O
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomTokenClassifier(model_name, num_labels)

    dataset = load_dataset("midas/inspec", "extraction")
    
    for item in dataset["train"]:
        text = item["document"]
        bio_labels = item["doc_bio_tags"]

        tokenized_inputs = tokenizer(text, 
                           is_split_into_words=True,
                           return_tensors="pt",
                           )

        labels = fix_labels(tokenized_inputs, bio_labels)
        
        # Forward pass (in training mode, labels are provided so the loss is computed)
        loss, logits, probs = model(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, labels=labels)
        print("Loss:", loss.item())
        print("Logits shape:", logits.shape)
        print("Probabilities shape:", probs.shape)
