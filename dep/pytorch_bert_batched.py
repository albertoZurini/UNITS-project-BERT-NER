# Taken from: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://arxiv.org/pdf/1910.08840

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, dim):
    max_score = torch.max(vec, dim=dim, keepdim=True)[0]
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score), dim=dim))

class BiLSTM_CRF(nn.Module):

    def __init__(self, model_name, tag_to_ix, embedding_dim, device):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device

        self.base_model = AutoModel.from_pretrained(model_name, reference_compile=True)
        hidden_dim = self.base_model.config.hidden_size
        self.hidden_dim = hidden_dim

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats, mask):
        batch_size, seq_len, tagset_size = feats.size()
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.0, device=self.device)
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.0
        forward_var = init_alphas

        for t in range(seq_len):
            feat_t = feats[:, t, :]
            mask_t = mask[:, t].unsqueeze(1)  # (batch_size, 1)
            forward_expanded = forward_var.unsqueeze(2)
            transitions = self.transitions.unsqueeze(0)
            scores = forward_expanded + transitions
            max_scores = torch.max(scores, dim=1, keepdim=True)[0]
            log_sum_exp_scores = max_scores + torch.log(torch.sum(torch.exp(scores - max_scores), dim=1, keepdim=True))
            log_sum_exp_scores = log_sum_exp_scores.squeeze(1) + feat_t
            forward_var = torch.where(mask_t.bool(), log_sum_exp_scores, forward_var)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0)
        alpha = log_sum_exp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, input_ids, attention_mask):
        # self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out = self.lstm(embeds, self.hidden)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        self.hidden = outputs.last_hidden_state

        # lstm_out = lstm_out.view(len(input_ids), self.hidden_dim)
        lstm_feats = self.hidden2tag(self.hidden)
        return lstm_feats

    def _score_sentence(self, feats, tags, mask):
        batch_size, seq_len = tags.size()
        start_tags = torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=self.device)
        tags_with_start = torch.cat([start_tags, tags], dim=1)
        score = torch.zeros(batch_size, device=self.device)

        for t in range(seq_len):
            current_tag = tags_with_start[:, t+1]
            prev_tag = tags_with_start[:, t]
            trans_score = self.transitions[current_tag, prev_tag]
            emit_score = feats[:, t].gather(1, current_tag.unsqueeze(1)).squeeze(1)
            valid = mask[:, t].float()
            score += (trans_score + emit_score) * valid

        last_tag = tags_with_start[:, -1]
        stop_score = self.transitions[self.tag_to_ix[STOP_TAG], last_tag]
        score += stop_score
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats[0]:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, input_ids, attention_mask, tags):
        feats = self._get_lstm_features(input_ids, attention_mask)
        mask = (attention_mask == 1)
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return (forward_score - gold_score).mean()

    def forward(self, input_ids, attention_mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(input_ids, attention_mask)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

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
        return 0 # [1, 0, 0]
    elif bio == "I":
        return 1 # [0, 1, 0]
    else:
        return 2 # [0, 0, 1]

def fix_labels(tokenized_inputs, bio_labels, tokenizer, debug=False):
    word_ids = tokenized_inputs.word_ids()
    new_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(2)  # Use 'O' for special tokens
        elif word_idx != prev_word_idx:
            new_labels.append(bio_labels[word_idx])
        else:
            label = bio_labels[word_idx]
            if label.startswith("B"):
                label = label.replace("B", "I")
            new_labels.append(label)
        prev_word_idx = word_idx

    new_labels = [bio_to_ohe(label) for label in new_labels]
    new_labels = torch.tensor(new_labels)
    return new_labels

def training_step(dataloader, model, optimizer, loss_fn, tokenizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss = model.neg_log_likelihood(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

class MyDataset(torch.utils.data.Dataset):
    """
    Class to store the tweet data as PyTorch Dataset
    """
    
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels
        
    def __getitem__(self, idx):

        return {
            "input_ids": torch.tensor(self.encodings[idx]),
            "attention_mask": torch.tensor(self.attention_masks[idx]),
            "labels": torch.tensor(self.labels[idx])
        }
    
    def __len__(self):
        return len(self.labels)

def collate(batch):
    ips = [item['input_ids'][0] for item in batch]
    attn = [item['attention_mask'][0] for item in batch]
    lb = [item['labels'][0] for item in batch]

    # Pad sequences to the same length
    ips_padded = torch.nn.utils.rnn.pad_sequence(ips, batch_first=True, padding_value=0)
    attn_padded = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    lb_padded = torch.nn.utils.rnn.pad_sequence(lb, batch_first=True, padding_value=-100)  # Common for loss masking

    return {
        'input_ids': ips_padded,
        'attention_mask': attn_padded,
        'labels': lb_padded
    }


# Example usage:
if __name__ == "__main__":
    model_name = "answerdotai/ModernBERT-base"  # or any other pretrained model
    num_labels = 3  # For example, if you have tags like B, I, and O
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used: {}.".format(device))

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 5
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    model = BiLSTM_CRF(model_name=model_name,
                       tag_to_ix=tag_to_ix, 
                       embedding_dim=EMBEDDING_DIM,
                       device=device
                       )
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
        label = fix_labels(tokenized_inputs[i], bio_labels, tokenizer)
        labels.append(label)

    # train_loader = DataLoader(dataset["train"], batch_size=10, shuffle=True)
    # tensorDataset = TensorDataset(input_ids, attention_mask, labels)
    tensorDataset = MyDataset(input_ids, attention_masks, labels)
    train_loader = DataLoader(tensorDataset, batch_size=24, shuffle=True, collate_fn=collate)

    for epoch in tqdm(range(10)):
        loss = training_step(train_loader, model, optimizer, loss_fn, tokenizer, device)
        print(loss)

    torch.save(model.state_dict(), 'model_weights.pth')
