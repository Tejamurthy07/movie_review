import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# Function to decode numerical reviews back to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocessing function: Convert words to numbers & pad sequence
def preprocess_text(text, max_length=500):  # Change to 500
    if not text.strip():  # Check if input is empty
        return None
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Pad the sequence to match the model input shape
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_length)
    
    return np.array(padded_review)  # Ensure it returns a valid NumPy array

# Streamlit Web App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    if preprocessed_input is not None:
        prediction = model.predict(preprocessed_input)[0][0]  # Extract single value
        sentiment = 'Positive' if prediction > 0.5 else 'Negative'

        st.write(f'*Sentiment:* {sentiment}')
        st.write(f'*Prediction Score:* {prediction:.4f}')
    else:
        st.write('Please enter a valid movie review before')