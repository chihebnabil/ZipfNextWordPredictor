import re
import random
import numpy as np
from collections import defaultdict, Counter

class ZipfNextWordPredictor:
    def __init__(self, corpus_texts=None, n_gram_size=3, zipf_exponent=1.0):
        """
        Initialize the next word predictor with Zipf's law parameters
        
        Args:
            corpus_texts: List of texts to build the model from
            n_gram_size: Size of n-grams to use (default: 3 for trigrams)
            zipf_exponent: Exponent for Zipf's law (default: 1.0)
        """
        self.n_gram_size = n_gram_size
        self.zipf_exponent = zipf_exponent
        
        # Initialize model components
        self.n_gram_model = defaultdict(lambda: defaultdict(int))
        self.corpus_freq = Counter()
        
        # Build model if corpus provided
        if corpus_texts:
            self.build_model(corpus_texts)
    
    def simple_sent_tokenize(self, text):
        """Simple sentence tokenizer that doesn't rely on NLTK"""
        # Split on sentence-ending punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
        # Filter out empty strings
        return [s for s in sentences if s]
    
    def preprocess(self, text):
        """Convert text to lowercase and extract words while preserving sentence structure"""
        # Keep original case for better readability in output
        sentences = self.simple_sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            # Extract words and punctuation
            words = re.findall(r'\b\w+\b|[.,!?;]', sentence)
            if words:
                processed_sentences.append(words)
        
        return processed_sentences
    
    def build_model(self, corpus_texts):
        """Build n-gram model and frequency distributions from corpus texts"""
        all_sentences = []
        
        # Process each text in the corpus
        for text in corpus_texts:
            sentences = self.preprocess(text)
            all_sentences.extend(sentences)
        
        # Build n-gram model and corpus frequencies
        for sentence in all_sentences:
            # Update word frequencies
            for word in sentence:
                self.corpus_freq[word.lower()] += 1
            
            # Build n-grams (up to n_gram_size)
            for n in range(2, self.n_gram_size + 1):
                for i in range(len(sentence) - n + 1):
                    context = tuple(w.lower() for w in sentence[i:i+n-1])
                    next_word = sentence[i+n-1].lower()
                    self.n_gram_model[context][next_word] += 1
    
    def zipf_probability(self, rank):
        """Calculate Zipf's probability for a given rank"""
        return 1.0 / (rank ** self.zipf_exponent)
    
    def predict_next_words(self, context, num_predictions=5, temperature=1.0):
        """
        Predict multiple possible next words based on context using Zipf's law
        
        Args:
            context: List or tuple of preceding words
            num_predictions: Number of predictions to return
            temperature: Controls randomness (higher = more random)
            
        Returns:
            List of (word, probability) tuples sorted by decreasing probability
        """
        # Convert context to lowercase and ensure it's a tuple
        if isinstance(context, str):
            # If context is a string, split it into words
            context = re.findall(r'\b\w+\b|[.,!?;]', context.lower())
        
        context = tuple(w.lower() for w in context[-self.n_gram_size+1:] if w)
        
        # Try with the longest context first, then back off to shorter contexts
        for n in range(len(context), 0, -1):
            current_context = context[-n:]
            next_words = self.n_gram_model.get(current_context, {})
            
            if next_words:
                # Sort words by frequency (descending)
                sorted_words = sorted(next_words.items(), key=lambda x: (-x[1], x[0]))
                words, counts = zip(*sorted_words)
                
                # Apply Zipf's law with temperature adjustment
                ranks = range(1, len(words)+1)
                probabilities = [self.zipf_probability(r) for r in ranks]
                
                # Adjust probabilities with temperature
                if temperature != 1.0:
                    probabilities = [p ** (1.0/temperature) for p in probabilities]
                
                total = sum(probabilities)
                normalized_probs = [p/total for p in probabilities]
                
                # Return top predictions with their probabilities
                result = list(zip(words, normalized_probs))
                return result[:num_predictions]
        
        # Fallback to corpus frequency if no n-grams found
        if self.corpus_freq:
            sorted_words = sorted(self.corpus_freq.items(), key=lambda x: (-x[1], x[0]))
            words, counts = zip(*sorted_words[:num_predictions*2])  # Get more candidates
            
            ranks = range(1, len(words)+1)
            probabilities = [self.zipf_probability(r) for r in ranks]
            
            # Apply temperature
            if temperature != 1.0:
                probabilities = [p ** (1.0/temperature) for p in probabilities]
            
            total = sum(probabilities)
            normalized_probs = [p/total for p in probabilities]
            
            result = list(zip(words, normalized_probs))
            return result[:num_predictions]
        
        return []  # No predictions available
    
    def predict_completion(self, input_text, num_words=3, temperature=1.2):
        """
        Complete the given input text with a specified number of words
        
        Args:
            input_text: The text to complete
            num_words: Number of words to generate
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Original text with completion appended
        """
        # Process the input text to get context
        words = re.findall(r'\b\w+\b|[.,!?;]', input_text)
        
        # Initialize result with the input text
        result = input_text
        
        # Generate specified number of words
        for _ in range(num_words):
            # Get context from the end of current text
            context = words[-self.n_gram_size+1:]
            
            # Predict next words
            predictions = self.predict_next_words(context, num_predictions=10, temperature=temperature)
            
            if not predictions:
                break
                
            # Choose next word based on probabilities
            words_list, probs_list = zip(*predictions)
            next_word = random.choices(words_list, weights=probs_list, k=1)[0]
            
            # Proper spacing for punctuation
            if next_word in ".,!?;":
                result += next_word
            else:
                # Add space if the last character isn't a space
                if result and not result[-1].isspace():
                    result += " "
                result += next_word
            
            # Update words list for next iteration
            words.append(next_word)
        
        return result


