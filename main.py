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


from colorama import Fore, Style, init
# Initialize colorama
init(autoreset=True)

# Example usage
if __name__ == "__main__":
    # Sample text for training the model
    sample_text = """
    Once upon a time in a far away land, there lived a young princess. She was known for her kindness and wisdom.
    The kingdom was prosperous under the rule of her father, the king. However, darkness loomed on the horizon.
    A wicked sorcerer cast a spell over the land, causing crops to fail and rivers to dry up.
    The princess decided to embark on a quest to break the curse. She gathered a small party of loyal friends.
    Together they journeyed through forests and mountains, facing many dangers along the way.
    They encountered magical creatures, some friendly and others hostile. The princess learned much from these encounters.
    After many trials and tribulations, they finally reached the sorcerer's tower. A great battle ensued.
    Using her wisdom and the help of her friends, the princess was able to defeat the sorcerer.
    The curse was lifted, and the land began to heal. The princess returned home to great celebration.
    Her father was so proud that he declared a holiday in her honor. And they all lived happily ever after.

    In another kingdom, a young farmer discovered a mysterious artifact while plowing his field.
    The artifact glowed with an eerie blue light whenever the moon was full.
    Word of this discovery spread quickly, and soon scholars from across the land came to study it.
    One scholar, an elderly woman with silver hair, recognized the artifact from ancient texts.
    She explained that it was a key to an ancient repository of knowledge, hidden beneath the mountains.
    The farmer, curious about his discovery, decided to accompany the scholar on an expedition.
    They gathered supplies and set out on the long journey to the mountains.
    Along the way, they faced bandits, harsh weather, and treacherous terrain.
    Finally reaching the mountains, they found the hidden entrance described in the texts.
    Using the artifact as a key, they unlocked the door to reveal countless scrolls and books.
    """

    # Create the predictor with the sample text
    predictor = ZipfNextWordPredictor(
        corpus_texts=[sample_text],
        n_gram_size=3,
        zipf_exponent=1.0
    )
    
    # Example 1: Get multiple next word predictions
    context = "Once upon a"
    predictions = predictor.predict_next_words(context, num_predictions=5, temperature=1.0)
    
    print(f"{Fore.CYAN}Context: '{context}'{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Top 5 predictions:{Style.RESET_ALL}")
    for word, prob in predictions:
        print(f"  {Fore.GREEN}'{word}'{Style.RESET_ALL} (probability: {Fore.MAGENTA}{prob:.4f}{Style.RESET_ALL})")
    
    # Example 2: Complete a sentence with different temperatures
    input_text = "The princess decided to"
    
    print(f"\n{Fore.CYAN}Input: {input_text}{Style.RESET_ALL}")
    
    # With lower temperature (more predictable)
    completion_low_temp = predictor.predict_completion(input_text, num_words=5, temperature=0.8)
    print(f"{Fore.YELLOW}Completion (low temperature):{Style.RESET_ALL} {Fore.GREEN}{completion_low_temp}{Style.RESET_ALL}")
    
    # With higher temperature (more creative)
    completion_high_temp = predictor.predict_completion(input_text, num_words=5, temperature=1.5)
    print(f"{Fore.YELLOW}Completion (high temperature):{Style.RESET_ALL} {Fore.GREEN}{completion_high_temp}{Style.RESET_ALL}")
    
    # Example 3: Interactive mode
    print(f"\n{Fore.CYAN}--- Interactive Mode ---{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type some text and the model will predict the next words.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' to quit.{Style.RESET_ALL}")
    
    while True:
        user_input = input(f"\n{Fore.CYAN}Your text: {Style.RESET_ALL}")
        if user_input.lower() == 'exit':
            break
            
        # Show multiple possible continuations
        predictions = predictor.predict_next_words(user_input, num_predictions=5, temperature=1.2)
        
        print(f"{Fore.YELLOW}Possible next words:{Style.RESET_ALL}")
        for word, prob in predictions:
            print(f"  {Fore.GREEN}'{word}'{Style.RESET_ALL} (probability: {Fore.MAGENTA}{prob:.4f}{Style.RESET_ALL})")
            
        # Complete the text
        completion = predictor.predict_completion(user_input, num_words=5, temperature=1.2)
        print(f"{Fore.YELLOW}Possible completion:{Style.RESET_ALL} {Fore.GREEN}{completion}{Style.RESET_ALL}")