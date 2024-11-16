import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.mixed_precision import set_global_policy
import pickle
import gradio as gr
from PIL import Image
import pytesseract
import threading
from difflib import get_close_matches
import time

# Mixed precision setup for maximum GPU utilization
set_global_policy('mixed_float16')
tf.config.experimental.enable_tensor_float_32_execution(False)

# Global Variables
MAX_WORDS = 10000
MAX_LEN = 100
BATCH_SIZE = 2048  # Larger batch size for full GPU utilization
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
model = None  # Placeholder for the trained model
tokenizer = None  # Placeholder for the tokenizer

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_names = [f"GPU {i}: {gpu.name}" for i, gpu in enumerate(gpus)]

# Function to configure GPU visibility
def set_gpu(gpu_indices):
    global gpus
    if not gpu_indices:
        print("Using CPU.")
        tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        try:
            # Set specific GPUs to be visible
            selected_gpus = [gpus[int(idx)] for idx in gpu_indices]
            tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {', '.join([gpu.name for gpu in selected_gpus])}")
        except RuntimeError as e:
            print(f"Error configuring GPU visibility: {e}")

# Function to find best column matches
def find_best_match(columns, target):
    matches = get_close_matches(target, columns, n=1, cutoff=0.5)
    return matches[0] if matches else None

# Preprocessing functions
def preprocess_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Optimize Data Loading with tf.data
def preprocess_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=50000)  # Utilize full RAM for shuffling
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Optimize loading
    return dataset

# Load model and tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")

# Training function with performance metrics
def train_model(file, gpu_choice, multi_gpu, epochs=5):
    try:
        # Set GPUs based on user selection
        if multi_gpu:
            print("Using all available GPUs with custom communication.")
            strategy = tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
            )
        else:
            # Convert selected GPU names to indices and configure GPUs
            gpu_indices = [gpu_names.index(gpu) for gpu in gpu_choice]
            set_gpu(gpu_indices)
            strategy = tf.distribute.get_strategy()

        # Load and preprocess data
        df = None
        encodings = ["utf-8", "ISO-8859-1", "utf-8-sig"]
        for encoding in encodings:
            try:
                df = pd.read_csv(file.name, encoding=encoding, header=None)  # Sentiment140 has no headers
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            return "Failed to read the CSV file. Please check the file format or encoding."

        # Check if the file matches the Sentiment140 format (6 columns)
        if len(df.columns) == 6:
            df.columns = ["label", "ids", "date", "flag", "user", "text"]
            df = df[["text", "label"]]
            df["label"] = df["label"].apply(lambda x: 1 if x == 4 else 0)
        else:
            columns = df.columns
            text_col = find_best_match(columns, "text")
            label_col = find_best_match(columns, "label")
            if not text_col or not label_col:
                return f"Unable to detect required columns. Found columns: {', '.join(columns)}."
            df = df.rename(columns={text_col: "text", label_col: "label"})

        # Tokenization
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(df["text"])
        sequences = tokenizer.texts_to_sequences(df["text"])
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        y = np.array(df["label"])

        # Split dataset
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = preprocess_dataset(X_train, y_train, BATCH_SIZE)
        val_dataset = preprocess_dataset(X_val, y_val, BATCH_SIZE)

        # Build and train model
        with strategy.scope():
            global model
            model = Sequential([
                Embedding(input_dim=MAX_WORDS, output_dim=256, input_length=MAX_LEN),
                Bidirectional(LSTM(512, return_sequences=True)),
                Bidirectional(LSTM(256)),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid', dtype='float32')
            ])

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            # Train model
            model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)

        # Save model and tokenizer
        model.save(MODEL_PATH)
        with open(TOKENIZER_PATH, 'wb') as f:
            pickle.dump(tokenizer, f)

        return "Training complete! Model and tokenizer saved."

    except Exception as e:
        return f"An error occurred during training: {str(e)}"

# Inference function
def analyze_input(input_text=None, input_image=None):
    global model, tokenizer  # Declare them as global

    if model is None or tokenizer is None:
        return "Model not trained yet.", "N/A"

    if input_image is not None:
        text = extract_text_from_image(input_image)
        if not text:
            return "No text detected in the image.", "N/A"
    elif input_text:
        text = input_text
    else:
        return "Please provide text or an image.", "N/A"

    padded_sequence = preprocess_text(text, tokenizer)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, f"Confidence: {prediction:.2f}"

# Gradio Interfaces
train_interface = gr.Interface(
    fn=train_model,
    inputs=[
        gr.File(label="Upload Training Dataset (CSV)"),
        gr.CheckboxGroup(gpu_names, label="Select GPUs to Use"),
        gr.Checkbox(label="Use All GPUs"),
        gr.Slider(1, 10, step=1, label="Number of Epochs"),
    ],
    outputs="text",
    title="Train Sentiment Model with GPU Selection",
    description="Upload a CSV file, select GPUs, and train the sentiment model."
)

inference_interface = gr.Interface(
    fn=analyze_input,
    inputs=[
        gr.Textbox(label="Enter Text", placeholder="Type your text here"),
        gr.Image(label="Upload Image", type="pil"),
    ],
    outputs=[
        gr.Text(label="Sentiment"),
        gr.Text(label="Confidence"),
    ],
    title="Multi-Modal Sentiment Analysis",
    description="Upload an image or input text to analyze sentiment."
)

# Launch Both Interfaces on Different Ports
def launch_training_interface():
    train_interface.launch(share=False, server_name="localhost", server_port=7861)

def launch_inference_interface():
    inference_interface.launch(share=True, server_name="localhost", server_port=7860)

if __name__ == "__main__":
    threading.Thread(target=launch_training_interface, daemon=True).start()
    threading.Thread(target=launch_inference_interface, daemon=True).start()
    input("Press Enter to exit...\n")
