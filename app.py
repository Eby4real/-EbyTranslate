from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Ensure model and tokenizer files exist
import os

if os.path.exists('model/hybrid_model.h5'):
    print("Model file found!")
else:
    print("Model file not found.")

if os.path.exists('model/english_tokenizer.pkl'):
    print("English tokenizer found!")
else:
    print("English tokenizer not found.")

if os.path.exists('model/french_tokenizer.pkl'):
    print("French tokenizer found!")
else:
    print("French tokenizer not found.")

def load_model_and_tokenizers():
    # Load the model
    hybrid_model = load_model('model/hybrid_model.h5')

    # Load the tokenizers
    with open("model/english_tokenizer.pkl", 'rb') as handle:
        english_tokenizer = pickle.load(handle)
    with open("model/french_tokenizer.pkl", 'rb') as handle:
        french_tokenizer = pickle.load(handle)

    return hybrid_model, english_tokenizer, french_tokenizer

# Load the models and tokenizers
model, english_tokenizer, french_tokenizer = load_model_and_tokenizers()

def preprocess_text(text, tokenizer, maxlen=55):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')
    return padded_sequence

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ''
    
    # Use axis=-1 to get the index of the highest predicted probability for each sample
    predicted_indices = np.argmax(logits, axis=-1)  # Get the index of the highest probability
    # Convert each index to its corresponding word
    translated_words = [index_to_words[prediction] for prediction in predicted_indices[0]]  # Only take the first sample
    return ' '.join(translated_words)


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Render index.html from the templates directory

@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        if not request.json or 'text' not in request.json:
            return jsonify({'error': 'No input text provided'}), 400

        input_text = request.json['text']
        preprocessed_text = preprocess_text(input_text, english_tokenizer)
        prediction = model.predict(preprocessed_text)
        translated_text = logits_to_text(prediction, french_tokenizer)

        return jsonify({'input_text': input_text, 'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
