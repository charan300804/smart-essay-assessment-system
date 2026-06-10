import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
from . import config
from . import data_loader
from . import preprocess
from . import model as model_module

def custom_kappa_metric(y_true, y_pred):
    # This is a bit complex to implement as a Keras metric directly due to tensor operations
    # So we usually evaluate it at the end of epoch or training
    pass

def train():
    print("Loading data...")
    try:
        df = data_loader.load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Filter for a specific essay set or use all?
    # Usually ASAP has different scales for different sets.
    # For simplicity, let's normalize scores or just train on Essay Set 1 for the demo.
    # Essay Set 1 ranges from 2 to 12.
    print("Filtering for Essay Set 1 (Argumentative) for this demo...")
    df = df[df['essay_set'] == 1]
    
    # Preprocess
    print("Preprocessing essays...")
    
    # Simple cleaning function needs to be applied to the 'essay' column
    cleaned_essays = [preprocess.clean_text(text) for text in df['essay']]
    
    # Tokenization and Padding
    # Note: We fit the tokenizer here
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_essays)
    
    # Save tokenizer for inference
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
        
    with open(os.path.join(config.MODELS_DIR, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(cleaned_essays)
    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    y = df['domain1_score'].values

    # Load Embeddings
    print("Loading embeddings...")
    embedding_matrix = preprocess.load_glove_embeddings(tokenizer)

    # Cross-validation setup (optional, but good for research)
    # For this demo, we'll do a simple Train/Validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

    # Create Model
    model = model_module.create_model(embedding_matrix)
    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(config.MODELS_DIR, 'best_model.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )

    # Evaluation
    print("\nEvaluating Model...")
    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred).flatten() # Round to nearest integer for Kappa
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    qwk = cohen_kappa_score(y_val, y_pred, weights='quadratic')
    
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"QWK: {qwk}")

    # Save final model
    model.save(os.path.join(config.MODELS_DIR, 'final_model.h5'))
    print("Training Complete.")

if __name__ == "__main__":
    import tensorflow as tf # Re-importing here just in case
    train()
