import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from . import config
import os
import pickle

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the input text: lowercasing, removing special characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
    # Optional: Remove stopwords. For essay scoring, sometimes structure matters, so removing stopwords might lose context.
    # We will keep it simple for now and keep stopwords or remove them based on performance.
    # stop_words = set(stopwords.words('english'))
    # words = word_tokenize(text)
    # text = " ".join([w for w in words if w not in stop_words])
    return text

def preprocess_essays(essays, tokenizer=None, fit_tokenizer=True):
    """
    Cleans, tokenizes, and pads a list of essays.
    """
    cleaned_essays = [clean_text(essay) for essay in essays]
    
    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(cleaned_essays)
        
        # Save tokenizer
        with open(os.path.join(config.MODELS_DIR, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif tokenizer is None:
         # Load tokenizer if not provided and not fitting
        if os.path.exists(os.path.join(config.MODELS_DIR, 'tokenizer.pickle')):
             with open(os.path.join(config.MODELS_DIR, 'tokenizer.pickle'), 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
             raise ValueError("Tokenizer must be provided or fitted.")

    sequences = tokenizer.texts_to_sequences(cleaned_essays)
    padded_sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def load_glove_embeddings(tokenizer):
    """
    Loads GloVe embeddings and creates an embedding matrix.
    """
    embeddings_index = {}
    if not os.path.exists(config.GLOVE_FILE):
        print(f"Warning: GloVe file not found at {config.GLOVE_FILE}. Using random embeddings.")
        return None

    with open(config.GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((config.VOCAB_SIZE, config.EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < config.VOCAB_SIZE:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
    return embedding_matrix

if __name__ == "__main__":
    # Test preprocessing
    sample_essays = ["This is a sample essay.", "Another good essay here."]
    padded, tok = preprocess_essays(sample_essays)
    print("Padded shape:", padded.shape)
