from book_recommender_v2 import RatingDataset, ContentDataset, UserDataset, FavouriteDataset, WeightedRatingModel, GradientDescentExpModel, Utils
import pandas as pd
import numpy as np

from fetch_data import (
    get_books_data,
    get_user_info_data,
    get_favourite_books_data,
    close_all_connections,
)

# --- Configuration ---
SENTENCE_TRANSFORMER_PATH = "preprocessed_data/sentence_transformer.pkl"

RATING_N_PATH = "preprocessed_data/rating_N.pkl"
RATING_DATASET_PATH = "preprocessed_data/rating_dataset.pkl"
RATING_COUNT_THRESHOLD_PATH = "preprocessed_data/rating_count_threshold.pkl"
RATING_AVG_MEAN_PATH = "preprocessed_data/rating_avg_mean.pkl"

CONTENT_N_PATH = "preprocessed_data/content_N.pkl"
CONTENT_INPUTS_PATH = "preprocessed_data/content_inputs.pkl"
CONTENT_IDS_PATH = "preprocessed_data/content_ids.pkl"
CONTENT_LABELS_PATH = "preprocessed_data/content_labels.pkl"
CONTENT_W_PATH = "preprocessed_data/content_W.pkl"

USER_N_PATH = "preprocessed_data/user_N.pkl"
USER_INPUTS_PATH = "preprocessed_data/user_inputs.pkl"
USER_IDS_PATH = "preprocessed_data/user_ids.pkl"
USER_LABELS_PATH = "preprocessed_data/user_labels.pkl"
USER_W_PATH = "preprocessed_data/user_W.pkl"

FAVOURITE_DATASET_PATH = "preprocessed_data/favourite_dataset.pkl"

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
            ]
        )

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
            ]
        )
        
        favourite_books_df = pd.DataFrame(
            favourite_books_data,
            columns=[
                "user_id",
                "book_id",
                "added_at"
            ]
        )

        return content_df, user_df, favourite_books_df

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None

# Load data from the database
print("Loading data from the database...")
content_df, user_df, favourite_df = load_data_from_database()

if content_df is None or user_df is None or favourite_df is None:
    raise Exception("Failed to load data from the database.")


# Preprocess and store embeddings
print("Preprocessing and storing embeddings...")
encoding_model = 'all-mpnet-base-v2'
scaler = 'min-max'

rating_dataset = RatingDataset(content_df)
print("Load Rating Dataset successfully")
content_dataset = ContentDataset(content_df, SENTENCE_TRANSFORMER_PATH, scaler)
print("Load Content Dataset successfully")
user_dataset = UserDataset(user_df, SENTENCE_TRANSFORMER_PATH, scaler)
print("Load User Dataset successfully")
favourite_dataset = FavouriteDataset(favourite_df)
print("Load Favourite Dataset successfully")


# Fit dataset and train models
rating_quantile_rate=0.9
rating_N = 10
rating_model = WeightedRatingModel(rating_quantile_rate, rating_N)
rating_model.fit(rating_dataset)

content_N=10
content_initial_W=None
content_lambda=np.array([1, 1, 2, 2, 1, 1, 1])
content_learning_rate=0.045
content_max_epoch=4
content_model = GradientDescentExpModel(content_lambda, content_learning_rate, content_max_epoch, content_N)
content_model.fit(content_dataset.get_inputs(), content_dataset.get_ids(), content_dataset.get_labels())

content_W = content_model.train(content_initial_W)
print("Content W:", content_W)

user_N=10
user_initial_W=None
user_lambda=np.array([2, 2, 1, 1, 1, 1])
user_learning_rate=0.06
user_max_epoch=160
user_model = GradientDescentExpModel(user_lambda, user_learning_rate, user_max_epoch, user_N)
user_model.fit(user_dataset.get_inputs(), user_dataset.get_ids(), user_dataset.get_labels())

user_W = user_model.train(user_initial_W)
print("User W:", user_W)


# Save Precomputed Data
print("Saving precomputed data...")
rating_model.save_model(RATING_N_PATH, RATING_DATASET_PATH, RATING_COUNT_THRESHOLD_PATH, RATING_AVG_MEAN_PATH)
content_model.save_model(CONTENT_N_PATH, CONTENT_INPUTS_PATH, CONTENT_IDS_PATH, CONTENT_LABELS_PATH, CONTENT_W_PATH)
user_model.save_model(USER_N_PATH, USER_INPUTS_PATH, USER_IDS_PATH, USER_LABELS_PATH, USER_W_PATH)
favourite_dataset.save_dataset(FAVOURITE_DATASET_PATH)
print("Preprocessing and data saving completed.")
close_all_connections()