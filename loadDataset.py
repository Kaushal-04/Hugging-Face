from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Fetch a review from the training set
review_number = 42
sample_review = dataset["train"][review_number]

# Access the text from the dataset key name
print(sample_review["text"][:450] + "...")

# Check sentiment label
if sample_review["label"] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
