from book_recommender import BookRecommender
import pandas as pd

# --- Configuration ---
SENTENCE_TRANSFORMER_PATH = (
    "./sentence_transformer.pkl"  # Path to your saved Sentence Transformer (optional)
)
CONTENT_DATASET_PATH = "./content_based_data.csv"
USER_DATASET_PATH = "./user_based_data.csv"
FAVORITE_BOOKS_PATH = "./favo_books.csv"

CONTENT_EMBEDDINGS_PATH = "content_embeddings.pkl"
CONTENT_LABELS_PATH = "content_labels.pkl"
CONTENT_WEIGHTS_PATH = "content_w.pkl"
CONTENT_PREPROCESSED_INPUT_PATH = "content_preprocessed_input.pkl"

USER_EMBEDDINGS_PATH = "user_embeddings.pkl"
USER_LABELS_PATH="user_labels.pkl"
USER_WEIGHTS_PATH="user_w.pkl"
USER_PREPROCESSED_INPUT_PATH="user_preprocessed_input.pkl"
        
FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH="favorite_preprocessed_input.pkl"


# --- Load Data and Preprocess ---
try:
    recommender = BookRecommender(sentence_transformer_path=SENTENCE_TRANSFORMER_PATH)
except Exception as e:
    print(f"Error loading Sentence Transformer: {e}")
    print("Loading Sentence Transformer from scratch...")
    recommender = BookRecommender()

try:
    content_df = pd.read_csv(CONTENT_DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the dataset file: {CONTENT_DATASET_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the dataset: {e}")

try:
    user_df = pd.read_csv(USER_DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the user dataset file: {USER_DATASET_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the user dataset: {e}")

try:
    favorite_books_df = pd.read_csv(FAVORITE_BOOKS_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the favorite books file: {FAVORITE_BOOKS_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the favorite books file: {e}")

# Preprocess and store embeddings
recommender.preprocess_and_store_book_embeddings(content_df)
recommender.preprocess_and_store_user_embeddings(user_df)
recommender.preprocess_and_store_favorite_books(favorite_books_df)
# Save Precomputed Data
recommender.save_precomputed_data(
    content_embeddings_path=CONTENT_EMBEDDINGS_PATH,
    content_labels_path=CONTENT_LABELS_PATH,
    content_weights_path=CONTENT_WEIGHTS_PATH,
    content_preprocessed_input_path=CONTENT_PREPROCESSED_INPUT_PATH,
    
    user_embeddings_path=USER_EMBEDDINGS_PATH,
    user_labels_path=USER_LABELS_PATH,
    user_weights_path=USER_WEIGHTS_PATH,
    user_preprocessed_input_path=USER_PREPROCESSED_INPUT_PATH,
    
    favorite_preprocessed_input_path=FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH,
)

print("Precomputed data saved successfully!")

# --- Additional Processing for Recommendations ---

# Personalized recommendations (based on preference)
user_id = 'USER-002'
