import torch
from collections import Counter

def bio_to_ohe(bio):
    if bio == "B":
        return 0
    elif bio == "I":
        return 1
    else:
        return 2

def ohe_to_bio(ohe):
    if ohe == 0:
        return "B"
    elif ohe == 1:
        return "I"
    else:
        return "O"

def fix_labels(tokenized_inputs, bio_labels, tokenizer, debug=False):
    word_ids = tokenized_inputs.word_ids()
    new_labels = []
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special token (e.g. [CLS], [SEP]) gets a dummy label.
            new_labels.append(-100)
        elif word_idx != prev_word_idx:
            # First token of a new word: assign its original BIO tag.
            new_labels.append(bio_labels[word_idx])
        else:
            # If the label starts with "B", change it to "I".
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

def training_step(dataset, model, optimizer, device):
    epoch_loss = 0
    for i, item in enumerate(dataset):
        model.zero_grad()
        batch_input_ids = item["input_ids"]
        batch_labels = item["labels"]
        batch_attention_mask = item["attention_mask"]

        for j in range(len(batch_input_ids)):
            input_ids = batch_input_ids[j]
            labels = batch_labels[j]
            attention_mask = batch_attention_mask[j]

            length = (attention_mask == 1).sum().item()
            input_ids = input_ids[:length].unsqueeze(0).to(device)
            labels = labels[:length].to(device)
            attention_mask = attention_mask[:length].unsqueeze(0).to(device)

            loss = model.neg_log_likelihood(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
    return epoch_loss

def print_f_score(dataset, tokenizer, device, model):
    TP = Counter()
    FP = Counter()
    FN = Counter()
    
    for sample in dataset:      
        # Tokenize the sample. Assumes the document is already split into words.
        tokenized_input = tokenizer(sample["document"], 
                                    is_split_into_words=True,
                                    return_tensors="pt")
        
        # Fix labels using the provided BIO tags.
        labels = fix_labels(tokenized_input, sample["doc_bio_tags"], tokenizer)
        
        # Move inputs to device.
        input_ids = tokenized_input["input_ids"].to(device)
        attention_mask = tokenized_input["attention_mask"].to(device)
        
        # Run inference.
        outputs = model.forward(input_ids, attention_mask)
                
        for true, pred in zip(outputs[1], labels[0]):
            # Calculate TP, FP, FN for each class
            pred = pred.item()
            if true == pred:
                TP[true] += 1
            else:
                FP[pred] += 1
                FN[true] += 1
            
    # Calculate precision, recall, and F1 score for each class
    precision = {}
    recall = {}
    f1_score = {}
    for label in [0, 1, 2]:
        precision[label] = TP[label] / (TP[label] + FP[label]) if (TP[label] + FP[label]) > 0 else 0
        recall[label] = TP[label] / (TP[label] + FN[label]) if (TP[label] + FN[label]) > 0 else 0
        f1_score[label] = (2 * precision[label] * recall[label] / (precision[label] + recall[label])
                            if (precision[label] + recall[label]) > 0 else 0)

    # Calculate Macro-Averaged F1 Score
    macro_f1 = sum(f1_score.values()) / len(f1_score) if f1_score else 0
    print("Macro-Averaged F1 Score:", macro_f1)

    # Output the results
    for label in sorted(f1_score):
        print(f"Class {label}: Precision={precision[label]:.2f}, Recall={recall[label]:.2f}, F1 Score={f1_score[label]:.2f}")