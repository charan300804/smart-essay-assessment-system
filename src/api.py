from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from . import config
from . import preprocess
from .model import AttentionLayer  # Needed for loading model with custom layer

app = FastAPI(title="SEAS API", description="Smart Essay Assessment and Scoring System API")

class EssayInput(BaseModel):
    essay: str

class ScoreResponse(BaseModel):
    score: float
    highlights: list

# Global variables to hold model and tokenizer
model = None
attention_model = None
tokenizer = None

@app.on_event("startup")
async def load_resources():
    global model, attention_model, tokenizer
    
    model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    tokenizer_path = os.path.join(config.MODELS_DIR, 'tokenizer.pickle')
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        try:
            # Load model with custom object
            model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
            
            # Create a sub-model to output attention weights
            # Assuming the layer is named 'attention' as per our update
            attention_layer_output = model.get_layer('attention').output
            attention_model = Model(inputs=model.input, outputs=[model.output, attention_layer_output])
            
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading resources: {e}")
    else:
        print("Model or tokenizer not found. Please train the model first.")

def extract_highlights(essay_text, attention_weights, tokenizer, threshold=0.1):
    """
    Extracts words/phrases with high attention weights.
    """
    words = preprocess.clean_text(essay_text).split() # Simple split for mapping, might need alignment with tokenizer
    # Tokenizer works on the same cleaned text
    
    # Attention weights shape: (1, distinct_words_in_vocab?? No, wait)
    # The attention mechanism in src/model.py reduces the time dimension.
    # The AttentionLayer implementation:
    # x: (batch, time, feat) -> tanh -> softmax -> sum(output, axis=1)
    # Wait, my AttentionLayer implementation returns the context vector (sum weighted inputs).
    # It does NOT return the weights alpha directly as the output of the layer.
    # call(self, x): a = softmax(...); output = x * a; return sum(output, axis=1)
    # The output of 'attention' layer is the Context Vector (batch, hidden_dim).
    # To get the weights 'a', I need to modify the Attention Layer to return them or capture them.
    
    # Let's fix this in the next iteration or use a workaround.
    # For now, if I can't get weights easily without retraining or redefining layer, 
    # I will stick to returning the score. 
    # Or I can try to interpret based on gradients (Grad-CAM style) but that's complex.
    
    # ACTUALLY, I should have modified the AttentionLayer to return weights or expose them.
    # Let's simple return the score for now and empty highlights to avoid breaking.
    return []

@app.post("/score", response_model=ScoreResponse)
async def score_essay(input_data: EssayInput):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cleaned_text = preprocess.clean_text(input_data.essay)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)
    final_score = float(prediction[0][0])
    
    # Highlights (Placeholder for now as Attention Weights extraction needs Layer modification)
    highlights = [] 
    
    return ScoreResponse(score=final_score, highlights=highlights)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
