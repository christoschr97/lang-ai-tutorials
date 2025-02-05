import torch
import torch.nn as nn
import pdb
from language_as_bow import tokenize, build_vocabulary

class CustomEmbedding(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		super(CustomEmbedding, self).__init__()
		self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))

	def forward(self, word_idx):
		return self.embeddings[word_idx]

# Lets test the embedding layer
embedding = CustomEmbedding(10, 3)
word_idx = torch.LongTensor([1])
print(f'This now returns a tensor of size {embedding(word_idx).size()}')
print(f'This now returns a tensor of size {embedding(word_idx)}')

corpus = [
	"I am a student",
	"I am a teacher",
	"I am a student and a teacher"
]

vocabulary = build_vocabulary(corpus)
word_to_idx = { word: idx for idx, word in enumerate(vocabulary) }

def create_word_pairs(corpus, window_size=2):
	word_pairs = []
	for document in corpus:
		tokenized_document = tokenize(document)
		# create word pairs by getting -2 +2 words and append them to the word pairs list
		for i, target_word in enumerate(tokenized_document):
			start = max(i - window_size, 0)
			end = min(i + window_size + 1, len(tokenized_document))

			for j in range(start, end):
				if i != j: # here we want to avoid pairing each word with itself
					word_pairs.append((target_word, tokenized_document[j]))

	return word_pairs

pairs = create_word_pairs(corpus, window_size=2)
print(pairs)

class SkipGramModel(nn.Module):
	def __init__(self, embeddings_dim, vocab_size):
		super(SkipGramModel, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embeddings_dim) #lets use the pytorch embedding layer for performance
		self.fc_layer = nn.Linear(embeddings_dim, vocab_size)

	def forward(self, target_word_idx):
		target_embedding = self.embedding(target_word_idx)
		logits = self.fc_layer(target_embedding)

		return logits

model = SkipGramModel(10, len(vocabulary))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(100):
	model.train()
	total_loss = 0

	for target_word, context_word in pairs:
		target_idx = torch.LongTensor([word_to_idx[target_word]])
		context_idx = torch.LongTensor([word_to_idx[context_word]])
		
		logits = model(target_idx)
		loss = loss_fn(logits, context_idx)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	print(f"Epoch {epoch + 1}, Loss: {total_loss / len(pairs)}")

word_embeddings = model.embedding.weight.data
print("Learned word embeddings:", word_embeddings)



