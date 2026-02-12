import torch
from transformers import AutoModel

class BERT_CRF(torch.nn.Module):
    def __init(self, model_name):
        super(BERT_CRF, self).__init__()

        self.base_model = AutoModel.from_pretrained(model_name, reference_compile=True)
        self.hidden_dim = self.base_model.config.hidden_size
