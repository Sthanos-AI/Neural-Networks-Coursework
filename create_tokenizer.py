from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

def train_custom_tokenizer(vocab_size=8000):
    dataset = load_dataset("ag_news")
    
    # Gather all texts
    texts = [t["text"] for split in ["train", "test"] for t in dataset[split]]
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Special tokens including class labels
    special_tokens = [
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
        "World", "Sports", "Business", "Sci/Tech"
    ]
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )
    
    # Save to disk
    save_path = Path("tokenizer_agnews")
    save_path.mkdir(exist_ok=True)
    tokenizer.save_model(str(save_path))
    
