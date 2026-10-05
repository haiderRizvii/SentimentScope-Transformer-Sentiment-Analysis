# SentimentScope: Transformer-Based Sentiment Analysis on IMDB

Udacity project: **Transformers for Movie Review Sentiment Analysis**.

As a Machine Learning Engineer at *CineScope*, the goal is to understand whether users feel positive or negative
about the movies they review, so the recommendation system can give more personalized suggestions. This repo has a
GPT-style transformer, **trained from scratch in PyTorch**, that classifies IMDB movie reviews as positive or negative.

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | **78.00 %** |
| Test accuracy (25,000 unseen reviews) | **76.24 %** |
| Project target | > 75 % ✅ |
| Untrained baseline (validation) | 50.40 % |
| Model size | 4.72 M parameters (20 MB checkpoint) |

## Repository contents

| File | Description |
|---|---|
| `SentimentScope.ipynb` | Full notebook with code and outputs: data loading, EDA, Dataset/DataLoader, model, training, evaluation, conclusion |
| `sentimentscope_model.pt` | Model checkpoint (weights + config + metrics). Gets more than 75 % test accuracy |
| `requirements.txt` | Python dependencies |

## Approach

1. **Data**: the 25,000 labelled training reviews (`aclImdb/train/pos|neg`) are loaded with a `load_dataset()` helper, shuffled and split 90/10 into training (22,500) and validation (2,500) sets. The 25,000 test reviews are used only for the final evaluation.
2. **Exploration**: descriptive statistics, the label distribution (perfectly balanced) and the review-length distribution (median 174 words, and 73.8 % of reviews are longer than 128 words).
3. **Tokenization and data loading**: subword tokenization with the `bert-base-uncased` tokenizer (padding and truncation to 128 tokens), inside a custom PyTorch `IMDBDataset`, batched by `DataLoader`s (batch size 32).
4. **Model (`DemoGPT`)**: token + positional embeddings, then 4 transformer blocks (4 heads × 32 dims, `d_embed` = 128, pre-LayerNorm, GELU feed-forward), then a final LayerNorm. Instead of predicting the next token, the outputs are **mean-pooled over the sequence (ignoring `[PAD]` tokens)** and passed to a **linear classification head** that outputs 2 logits. Softmax turns these into P(negative) and P(positive).
5. **Training**: cross-entropy loss, AdamW (lr = 3e-4), 4 epochs, gradient clipping. Validation loss and accuracy are tracked every epoch and the weights with the best validation accuracy are kept.
6. **Evaluation**: `calculate_accuracy()` on the held-out test set.

### Training curve

| Epoch | Train loss | Val loss | Val accuracy |
|---|---|---|---|
| 1 | 0.626 | 0.540 | 71.96 % |
| 2 | 0.501 | 0.488 | 76.52 % |
| 3 | 0.432 | 0.491 | 76.80 % |
| 4 | 0.378 | 0.474 | 78.00 % |

## Inference

```python
predictor = SentimentPredictor("sentimentscope_model.pt")   # class defined in the notebook
predictor.predict(["An absolute masterpiece!", "What a waste of two hours."])
# -> [{'label': 'positive', 'p_positive': ...}, {'label': 'negative', 'p_negative': ...}]
```

## How to run

```bash
pip install -r requirements.txt
# Download aclImdb_v1.tar.gz from https://ai.stanford.edu/~amaas/data/sentiment/ next to the notebook
jupyter notebook SentimentScope.ipynb
```

A GPU is recommended. On a 2-core CPU, a full run takes about 25 minutes.

## Key takeaways

- A small transformer trained from scratch beats the 75 % target (76.24 % test accuracy). The architecture used for text generation carries over to classification once the next-token head is replaced with pooling and a classifier.
- Excluding padding from mean pooling matters, because many reviews are shorter than the context window.
- Truncating reviews to 128 tokens is the main limitation, since the verdict often comes at the end of a review. Longer contexts, head + tail truncation, bidirectional attention, or fine-tuning a pre-trained encoder (e.g. DistilBERT, which reaches > 90 %) are the next steps.
