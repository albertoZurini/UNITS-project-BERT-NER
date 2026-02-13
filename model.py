import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F

START_TAG = "<START>"


def argmax(vec):
    # Return the argmax as a python int.
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm.
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BERT_CRF(nn.Module):
    def __init__(self, model_name, tag_to_idx):
        super(BERT_CRF, self).__init__()

        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        # Load BERT
        self.base_model = AutoModel.from_pretrained(model_name, reference_compile=True)
        self.hidden_dim = self.base_model.config.hidden_size

        # Maps the output of the LSTM into tag space.
        self.hidden2hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # Enforce constraints: never transfer to START
        self.transitions.data[self.tag_to_idx[START_TAG], :] = -10000

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device: torch.device):
        self.base_model.to(device)
        super().to(device)
        return self

    def _get_features(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        self.hidden = outputs.last_hidden_state

        features = self.hidden2hidden(self.hidden)
        features = F.relu(features)
        features = self.hidden2tag(features)
        return features

    def _forward_alg(self, feats, mask):
        """
        Calculate the partition function (log-sum-exp of all possible paths)
        in a vectorized manner for the batch.
        """
        batch_size, seq_len, tag_dim = feats.shape

        # Initialize alpha (forward variables)
        # Shape: (Batch_Size, Tag_Dim)
        # The START_TAG is assigned a score of 0., meaning the sequence must start from this tag.
        # All other tags still have -10000., making them nearly impossible as initial tags.
        alpha = torch.full((batch_size, tag_dim), -10000.0, device=self.device)
        alpha[:, self.tag_to_idx[START_TAG]] = 0.0

        # Transpose transitions to match the broadcasting shape later
        # We need (To, From), but for broadcasting usually (From, To) is easier
        # depending on how we sum.
        # Your definition: transitions[i, j] is To(i) From(j).

        # Loop through the sequence
        for t in range(seq_len):
            # feats_t: (Batch_Size, Tag_Dim) -> Emission scores at step t
            feats_t = feats[:, t, :]

            # mask_t: (Batch_Size) -> 1 if valid token, 0 if padding
            mask_t = mask[:, t].unsqueeze(1)

            # Broadcasting to calculate scores for all transitions at once:
            # alpha: (Batch, From, 1)
            # feats_t: (Batch, 1, To)
            # transitions: (1, To, From) (Wait, your definition is To, From)

            # Let's align dimensions:
            # alpha_prev: (Batch, From_Tags, 1)
            alpha_prev = alpha.unsqueeze(2)

            # emission: (Batch, 1, To_Tags)
            emission = feats_t.unsqueeze(1)

            # transitions: (1, From_Tags, To_Tags)
            # Note: self.transitions is (To, From). We need (From, To) for the addition
            # or we transpose the logic. Let's stick to your param definition:
            # transitions[next_tag, prev_tag]
            trans = self.transitions.T.unsqueeze(0)  # Shape (1, From, To)

            # Score: (Batch, From, To)
            next_tag_var = alpha_prev + trans + emission

            # LogSumExp across the "From" dimension (dim=1) to get the score for reaching "To"
            new_alpha = torch.logsumexp(next_tag_var, dim=1)

            # Masking:
            # If the token is valid (mask=1), we update alpha.
            # If the token is padding (mask=0), we keep the old alpha (effectively skipping this step).
            alpha = torch.where(mask_t > 0, new_alpha, alpha)

        # Final LogSumExp to get the total score for the sequence
        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, feats, tags, mask):
        """
        Calculates the score of the ground truth path (numerator in log-likelihood).
        This is the log-equivalent for equation 17.26 from the textbook.
        """
        batch_size, seq_len, _ = feats.shape

        # Initialize score
        score = torch.zeros(batch_size, device=self.device)

        # We can iterate or vectorize. Since we need to look up specific transitions,
        # a loop over seq_len with gathering is usually efficient enough and readable.

        for t in range(seq_len-1):
            # Current tag (To) and Previous tag (From)
            current_tags = tags[:, t + 1]  # (Batch)
            prev_tags = tags[:, t]  # (Batch)

            # Emission Score: feats[batch, t, current_tag]
            # gather expects index to have same dims as input
            emission = feats[:, t, :].gather(1, current_tags.unsqueeze(1)).squeeze(1)

            # Transition Score: transitions[current_tag, prev_tag]
            transition = self.transitions[current_tags, prev_tags]

            # Add to score, but only if mask is active
            step_score = emission + transition

            # mask[:, t] is 1 for valid, 0 for pad
            score = score + step_score * mask[:, t]

        return score

    def _viterbi_decode(self, feats, mask):
        """
        Decodes the best path. Loops over batch (standard for inference).
        """
        batch_size, seq_len, _ = feats.shape
        result_paths = []
        result_scores = []

        # Iterate over each sample in the batch
        for b in range(batch_size):
            # Get features for this sample
            # We must shorten the sequence based on the mask to avoid decoding padding
            valid_len = int(mask[b].sum().item())
            sample_feats = feats[b, :valid_len, :]
            
            # Initialize viterbi variables
            backpointers = []
            
            # Initialize forward variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.0, device=self.device)
            init_vvars[0][self.tag_to_idx[START_TAG]] = 0
            
            forward_var = init_vvars
            
            for feat in sample_feats:
                bptrs_t = [] 
                viterbivars_t = [] 

                for next_tag in range(self.tagset_size):
                    # transitions is (To, From), forward_var is (1, From)
                    # next_tag_var[i] = score(trans i -> next) + score(path to i)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Termination
            best_tag_id = argmax(forward_var)
            path_score = forward_var[0][best_tag_id]
            
            # Follow backpointers
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            
            # Pop off the start tag (implicit) and reverse
            start = best_path.pop() 
            best_path.reverse()
            
            result_scores.append(path_score)
            result_paths.append(best_path)
            
        return result_scores, result_paths


    def neg_log_likelihood(self, input_ids, attention_mask, tags):
        # From the paper https://arxiv.org/pdf/1910.08840
        # Formula (9)
        # p(y|f) = ( exp(s(f, y)) ) / (sum { exp(s(f, y')) })
        # Numerator is the score calculated from the ground truth
        # Denominator is the score calculated from the model's output
        # Since we are computing the log likelihood, we are computing
        # -log(p(y|f)) = -log(exp(..) / sum(..)) = -log(exp(..)) - (-log(sum{..}))
        # exp(..) is _score_sentence
        # sum{..} is _forward_alg
        feats = self._get_features(input_ids, attention_mask)
        forward_score = self._forward_alg(feats, attention_mask)
        gold_score = self._score_sentence(feats, tags, attention_mask)
        
        loss = forward_score - gold_score
        
        return loss.mean()

    def forward(
        self, input_ids, attention_mask
    ):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_features(input_ids, attention_mask)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats, attention_mask)
        return score, tag_seq

    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.base_model.named_parameters():
            param[1].requires_grad = False

    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.base_model.named_parameters():
            param[1].requires_grad = True
