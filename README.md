# BERT-based Named Entity Recognition (NER)

Dataset: https://huggingface.co/datasets/midas/inspec

## What is Named Entity Recognition?

Named Entity Recognition (NER) is a core task in Natural Language Processing (NLP) that identifies spans of text referring to real-world objects and classifies them into predefined categories such as people, organizations, locations, and more.

As defined in *Speech and Language Processing* (3rd ed.) by Jurafsky & Martin ([Chapter 17](https://web.stanford.edu/~jurafsky/slp3/17.pdf)):

> "Named entity recognition (NER) is the task of finding spans of text that constitute proper names, and labeling the type of the entity."

The term *named entity* is used broadly. While it originally referred strictly to proper names, modern NER systems also tag temporal expressions (dates, times), numerical expressions (money, percentages), and other domain-specific categories. Jurafsky & Martin note:

> "The term 'named entity' is due to [Grishman and Sundheim (1996)] and was originally intended to refer to the kinds of entities that are referred to by name, like people, places, and organizations. But the term has come to include entities that aren't strictly named, like dates and numerical expressions."

### Standard Entity Types

The four entity types from the CoNLL-2003 benchmark — widely used in NER research — are:

| Tag | Category | Examples |
|-----|----------|---------|
| `PER` | Person | *Albert Einstein*, *Marie Curie* |
| `ORG` | Organization | *United Nations*, *Apple Inc.* |
| `LOC` | Location | *Paris*, *Mount Everest* |
| `MISC` | Miscellaneous | *French*, *World Cup* |

### Sequence Labeling and BIO Tagging

NER is framed as a **sequence labeling** problem: every token in a sentence is assigned a label. The standard encoding is the **BIO scheme** (also called IOB):

- **B-TYPE** — the token is the *beginning* of an entity of the given type
- **I-TYPE** — the token is *inside* a continuing entity
- **O** — the token is *outside* any named entity

For example, the sentence *"Marie Curie worked at the University of Paris"* would be labeled:

```
Marie      → B-PER
Curie      → I-PER
worked     → O
at         → O
the        → O
University → B-ORG
of         → I-ORG
Paris      → I-ORG
```

Jurafsky & Martin describe why this matters:

> "NER is part of a larger class of problems known as sequence labeling, in which we assign a label to each token in a sequence. ... The most common sequence labeling approach for NER is based on the BIO encoding."

### Why NER Matters

NER is a foundational step in many downstream NLP pipelines. Jurafsky & Martin highlight its role in information extraction:

> "NER is one component in a number of NLP systems that extract factual information from text, enabling tasks like question answering, knowledge base construction, and event detection."

---

## This Project

This project implements a BERT-based NER model trained on the [Inspec dataset](https://huggingface.co/datasets/midas/inspec), which contains scientific abstracts annotated with keyphrases. BERT's bidirectional transformer encoder produces rich, context-sensitive token embeddings that are fed into a sequence labeling head (optionally with a CRF layer) to predict BIO tags.

## Parallel Code Performance

With the code from [this commit](https://github.com/albertoZurini/UNITS-project-BERT-NER/blob/8b809baf178dcc365c6e408b6dc2fc02b3782471/model.py), a whole epoch took 6 minutes. With the parallel-enabled execution, using a batch size of 24, the epoch duration went down to 45s.

---

*NER definitions and quotes from: Jurafsky, D. & Martin, J.H. (2024). Speech and Language Processing (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/*
