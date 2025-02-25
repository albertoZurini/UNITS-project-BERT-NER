import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tabulate import tabulate
from collections import Counter
import pickle

from model import BiLSTM_CRF, START_TAG, STOP_TAG
from utils import fix_labels, ohe_to_bio

def main():
    # Configuration.
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device used: {device}")

    EMBEDDING_DIM = 5
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    # Load the model.
    model = BiLSTM_CRF(model_name=model_name,
                       tag_to_ix=tag_to_ix, 
                       embedding_dim=EMBEDDING_DIM,
                       device=device)
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.freeze_bert()
    model.to(device)
    
    benchmarks = []
    
    for _split in ["train", "test"]:
      # Load the dataset directly from the datasets library.
      dataset = load_dataset("midas/inspec", "extraction", split=_split)
      
      TP = Counter()
      FP = Counter()
      FN = Counter()
      
      
      for sample in dataset:      
        # Tokenize the sample. Assumes the document is already split into words.
        
        sample = dataset[150]
        
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
        
        table_data = [
            [
                tokenizer.decode(input_ids[0][i]), 
                outputs[1][i], 
                ohe_to_bio(outputs[1][i]), 
                labels[0][i].item()  # Convert tensor to python int.
            ]
            for i in range(len(outputs[1]))
        ]
        
        headers = ["Token", "Output", "BIO Tag", "Label"]
        _out = tabulate(table_data, headers=headers, tablefmt="html")
        with open("table.html", "w") as f:
          f.write(_out)
        return
        
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
          
      benchmarks.append({
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "type": _split
      })
      
    with open("benchmark.pkl", "wb") as f:
      pickle.dump(benchmarks, f)

if __name__ == "__main__":
    main()
