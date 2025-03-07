# Zipf's Law Next Word Predictor

An experimental implementation of text prediction using Zipf's law to understand fundamental linguistic patterns that underlie modern language models.

## Overview

This project explores how Zipf's law - a fundamental principle in linguistics stating that the frequency of any word is inversely proportional to its rank in the frequency table - can be used for text prediction. While modern LLMs use sophisticated neural architectures, this experiment strips away the complexity to examine how basic statistical patterns contribute to text prediction.

## Key Features

- N-gram based text analysis with Zipf's law probability distribution
- Adjustable temperature for controlling prediction randomness
- Context-aware word prediction with frequency-based backoff
- Interactive text completion mode

## Theory and Implementation

The predictor combines:
1. N-gram model for capturing local context
2. Zipf's law for word frequency distribution (P(word) ∝ 1/rank^s)
3. Temperature-based sampling for controlling prediction diversity

## Significance

This implementation helps understand:
- The role of statistical patterns in language modeling
- How Zipf's law relates to modern LLM prediction patterns
- Baseline performance of statistical approaches vs neural models

## Usage

```python
predictor = ZipfNextWordPredictor(
    corpus_texts=[your_text],
    n_gram_size=3,
    zipf_exponent=1.0
)

# Get word predictions
predictions = predictor.predict_next_words("Once upon a")

# Complete a sentence
completion = predictor.predict_completion("The story begins", num_words=5)
```

## Limitations

- No semantic understanding
- Limited to patterns present in training corpus
- Lacks long-range dependencies
- Simple statistical approach vs modern attention mechanisms

## Experimental Insights

This project demonstrates how fundamental linguistic patterns captured by Zipf's law contribute to the statistical foundation of text prediction, providing insights into one aspect of how modern LLMs work at a basic level.