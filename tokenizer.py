# Load a Tokenizer
from transformers import BertTokenizer

# initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # uncased means not specified upper or lower case

# see how many tokens are in the vocabulary
print(tokenizer.vocab_size)

tokens = tokenizer.tokenize("My name is Kaushal Kumar") # Tokenize the sentence
print(tokens)

print(tokenizer.convert_tokens_to_ids(tokens)) # Show the token ids assigned to each token

tokens = tokenizer.tokenize("I am learning Generative AI") # Tokenize the sentence
print(tokens)

print(tokenizer.convert_tokens_to_ids(tokens)) # Show the token ids assigned to each token

# ['i', 'am', 'learning', 'genera', '##tive', 'ai']
# [1045, 2572, 4083, 11416, 6024, 9932]