In the Skip-Gram model of Word2Vec, the training process involves predicting context words given a target word. This is the opposite of the Continuous Bag of Words (CBOW) model, which predicts a target word from context words. In this setup:

- **Inputs**: The input is the target word.
- **Labels (Outputs)**: The labels are the context words surrounding the target word within a specified window size.

### Inputs and Labels in Skip-Gram

1. **Inputs (Target Word)**:
   - The target word is the center word in the window of words being considered. For each target word, the model tries to predict the words surrounding it (context words).

2. **Labels (Context Words)**:
   - The context words are the words that appear within a predefined window size around the target word. For instance, if the window size is 2, the context words for a target word would be the two words before and the two words after the target word.

### Example

Suppose we have a sentence: "The quick brown fox jumps over the lazy dog".

If the target word is "fox" and the window size is 2, the context words would be "quick", "brown", "jumps", and "over".

### Generating Training Data

To generate training data, we slide a window over the text to extract pairs of target and context words. For example:

- Sentence: "The quick brown fox jumps over the lazy dog"
- Window size: 2

| Target (Input) | Context (Label)      |
|----------------|-----------------------|
| quick          | The, brown            |
| brown          | The, quick, fox       |
| fox            | quick, brown, jumps, over |
| jumps          | brown, fox, over, the |
| over           | fox, jumps, the, lazy |
| the            | jumps, over, lazy, dog|
| lazy           | over, the, dog        |

### PyTorch Implementation

Here is an example of how to prepare inputs and labels for training a Skip-Gram model in PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from random import choice

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog".split()

# Building the vocabulary and word to index mapping
vocab = set(sentence)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Window size
window_size = 2

# Generate training pairs
def generate_pairs(sentence, window_size):
    pairs = []
    for i, target_word in enumerate(sentence):
        start = max(0, i - window_size)
        end = min(len(sentence), i + window_size + 1)
        context_words = [sentence[j] for j in range(start, end) if j != i]
        pairs.extend((target_word, context_word) for context_word in context_words)
    return pairs

pairs = generate_pairs(sentence, window_size)

# Convert words to indices
inputs = torch.tensor([word_to_idx[pair[0]] for pair in pairs], dtype=torch.long)
labels = torch.tensor([word_to_idx[pair[1]] for pair in pairs], dtype=torch.long)

# Skip-Gram Model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, input_word):
        return self.embeddings(input_word)

# Hyperparameters
embedding_dim = 100
vocab_size = len(vocab)

# Initialize model, loss function, and optimizer
model = SkipGramModel(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    total_loss = 0
    for input_word, target_word in zip(inputs, labels):
        model.zero_grad()
        output = model(input_word)
        loss = loss_function(output.unsqueeze(0), target_word.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")
```

### Conclusion

In the Skip-Gram model, the inputs are the target words, and the labels are the context words surrounding each target word within a specified window size. By training on these input-label pairs, the model learns to predict context words given a target word, resulting in meaningful word embeddings.