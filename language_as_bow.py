import re
import pdb

def tokenize(document):
    s = document.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = s.split()
    return s

def build_vocabulary(documents):
    vocabulary = set()
    for document in documents:
        tokenized_document = tokenize(document)
        vocabulary.update(tokenized_document)
    return vocabulary

def find_word_token_index_in_vocabulary(token, vocabulary):
    return list(vocabulary).index(token)

def build_bow_representation(document, vocabulary):
    bow_representation = [0] * len(vocabulary)
    tokenized_document = tokenize(document)
    for token in tokenized_document:
        index = find_word_token_index_in_vocabulary(token, vocabulary)
        bow_representation[index] += 1
    
    return bow_representation


def __main__():
    documents = [
        "I am a student",
        "I am a teacher",
        "I am a student and a teacher"
    ]

    vocabulary = build_vocabulary(documents)
    print(vocabulary)

    bow_representation = build_bow_representation(documents[0], vocabulary)
    print(bow_representation)

    bow_representation = build_bow_representation(documents[1], vocabulary)
    print(bow_representation)

if __name__ == "__main__":
    __main__()