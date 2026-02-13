Dataset: https://huggingface.co/datasets/midas/inspec

# Parallel code performance

With the code from [this commit](https://github.com/albertoZurini/UNITS-project-BERT-NER/blob/8b809baf178dcc365c6e408b6dc2fc02b3782471/model.py), a whole epoch took 6 minutes. With the parallel-enabled execution, using a batch size of 24, the epoch duration went down to 45s.

