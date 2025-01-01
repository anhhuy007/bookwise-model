from book_recommender import BookRecommender
import pandas as pd
from fetch_data import (
    get_books_data,
    get_user_info_data,
    get_favourite_books_data,
    close_all_connections,
)

# --- Configuration ---
# Paths to your precomputed data files
SENTENCE_TRANSFORMER_PATH = "preprocessed_data/sentence_transformer.pkl"
CONTENT_EMBEDDINGS_PATH = "preprocessed_data/content_embeddings.pkl"
CONTENT_LABELS_PATH = "preprocessed_data/content_labels.pkl"
CONTENT_WEIGHTS_PATH = "preprocessed_data/content_w.pkl"
CONTENT_PREPROCESSED_INPUT_PATH = "preprocessed_data/content_preprocessed_input.pkl"
USER_EMBEDDINGS_PATH = "preprocessed_data/user_embeddings.pkl"
USER_LABELS_PATH = "preprocessed_data/user_labels.pkl"
USER_WEIGHTS_PATH = "preprocessed_data/user_w.pkl"
USER_PREPROCESSED_INPUT_PATH = "preprocessed_data/user_preprocessed_input.pkl"
FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH = (
    "preprocessed_data/favorite_preprocessed_input.pkl"
)

# --- Load Data from Database ---
def load_data_from_database():
    """Loads data from the database and returns pandas DataFrames."""
    try:
        books_data = get_books_data()
        user_info_data = get_user_info_data()
        favourite_books_data = get_favourite_books_data()
        
        print(f"Fetch data from database: {len(books_data)} books, {len(user_info_data)} users, {len(favourite_books_data)} favorite books")

        # Convert to pandas DataFrames
        content_df = pd.DataFrame(
            books_data,
            columns=[
                "id",
                "isbn13",
                "title",
                "authors",
                "published_date",
                "page_count",
                "category",
                "language",
                "avg_rating",
                "rating_count",
                "img_url",
                "preview_url",
                "description",
            ],
        )
        # Handle published_date (extract year)
        content_df["published_date"] = pd.to_datetime(
            content_df["published_date"], errors="coerce"
        ).dt.year.fillna(0).astype(int)  

        # Ensure data type consistency
        content_df["avg_rating"] = content_df["avg_rating"].astype(float)
        content_df["rating_count"] = content_df["rating_count"].astype(int)
        
        user_df = pd.DataFrame(
            user_info_data,
            columns=[
                "id",
                "gender",
                "dob",
                "university",
                "faculty",
                "age",
                "language",
                "factor",
                "goal",
            ],
        )
        
        # 
        favorite_books_df = pd.DataFrame(
            favourite_books_data, columns=["user_id", "book_id", "added_at"]
        )

        return content_df, user_df, favorite_books_df

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None


# --- Load and Preprocess Data ---
try:
    recommender = BookRecommender(sentence_transformer_path=SENTENCE_TRANSFORMER_PATH)
except Exception as e:
    print(f"Error loading Sentence Transformer: {e}")
    print("Loading Sentence Transformer from scratch...")
    recommender = BookRecommender()

# Load data from the database
print("Loading data from the database...")
content_df, user_df, favorite_books_df = load_data_from_database()

if content_df is None or user_df is None or favorite_books_df is None:
    raise Exception("Failed to load data from the database.")

# Preprocess and store embeddings
print("Preprocessing and storing embeddings...")
recommender.preprocess_and_store_book_embeddings(content_df)
recommender.preprocess_and_store_user_embeddings(user_df)
recommender.preprocess_and_store_favorite_books(favorite_books_df)

# Save Precomputed Data
print("Saving precomputed data...")
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

print("Preprocessing and data saving completed.")
close_all_connections()
