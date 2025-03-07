from ZipfNextWordPredictor import ZipfNextWordPredictor
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