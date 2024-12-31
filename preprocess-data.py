from book_recommender import BookRecommender
import pandas as pd

# --- Configuration ---
SENTENCE_TRANSFORMER_PATH = (
    "./sentence_transformer.pkl"  # Path to your saved Sentence Transformer (optional)
)
DATASET_PATH = "./content_based_data.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
LABELS_PATH = "labels.pkl"
WEIGHTS_PATH = "content_w.pkl"
PREPROCESSED_INPUT_PATH = "preprocessed_input.pkl"

# --- Load Data and Preprocess ---
try:
    recommender = BookRecommender(sentence_transformer_path=SENTENCE_TRANSFORMER_PATH)
except Exception as e:
    print(f"Error loading Sentence Transformer: {e}")
    print("Loading Sentence Transformer from scratch...")
    recommender = BookRecommender()

try:
    content_df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the dataset file: {DATASET_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the dataset: {e}")

recommender.preprocess_and_store_embeddings(content_df)

# --- Save Precomputed Data ---
recommender.save_precomputed_data(
    embeddings_path=EMBEDDINGS_PATH,
    labels_path=LABELS_PATH,
    weights_path=WEIGHTS_PATH,
    preprocessed_input_path=PREPROCESSED_INPUT_PATH,
)

print("Precomputed data saved successfully!")
