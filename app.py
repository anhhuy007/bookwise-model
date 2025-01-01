from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Any
from book_recommender import BookRecommender
from fetch_data import get_books_data, get_user_info_data, get_favourite_books_data

# --- Configuration and Model Loading ---
app = FastAPI()

# Paths to your precomputed data files
CONTENT_EMBEDDINGS_PATH = "./preprocessed_data/content_embeddings.pkl"
CONTENT_LABELS_PATH = "./preprocessed_data/content_labels.pkl"
COTENT_WEIGHTS_PATH = "./preprocessed_data/content_w.pkl"  #
CONTENT_PREPROCESSED_INPUT_PATH = "./preprocessed_data/content_preprocessed_input.pkl"  #
USER_EMBEDDINGS_PATH = "./preprocessed_data/user_embeddings.pkl"
USER_LABELS_PATH = "./preprocessed_data/user_labels.pkl"
USER_WEIGHTS_PATH = "./preprocessed_data/user_w.pkl"
USER_PREPROCESSED_INPUT_PATH = "./preprocessed_data/user_preprocessed_input.pkl"
FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH = "./preprocessed_data/favorite_preprocessed_input.pkl"
SENTENCE_TRANSFORMER_PATH = "./preprocessed_data/sentence_transformer.pkl"

# Paths to your dataset files
CONTENT_DATASET_PATH = "./dataset/content_based_data.csv"
USER_DATASET_PATH = "./dataset/user_based_data.csv"
FAVORITE_BOOKS_PATH = "./dataset/favo_books.csv"

try:
    recommender = BookRecommender(sentence_transformer_path=SENTENCE_TRANSFORMER_PATH)
    recommender.load_precomputed_data(
        # Load content data
        content_embeddings_path=CONTENT_EMBEDDINGS_PATH,
        content_labels_path=CONTENT_LABELS_PATH,
        content_weights_path=COTENT_WEIGHTS_PATH,
        content_preprocessed_input_path=CONTENT_PREPROCESSED_INPUT_PATH,
        user_embeddings_path=USER_EMBEDDINGS_PATH,
        user_labels_path=USER_LABELS_PATH,
        user_weights_path=USER_WEIGHTS_PATH,
        user_preprocessed_input_path=USER_PREPROCESSED_INPUT_PATH,
        favorite_preprocessed_input_path=FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH,
    )
except Exception as e:
    print(f"Error loading precomputed data or Sentence Transformer: {e}")
    raise Exception("Error loading precomputed data or Sentence Transformer")

try:
    content_df = pd.read_csv(CONTENT_DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the dataset file: {CONTENT_DATASET_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the dataset: {e}")

try:
    user_df = pd.read_csv(USER_DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find the user dataset file: {USER_DATASET_PATH}"
    )
except Exception as e:
    raise Exception(f"An error occurred while loading the user dataset: {e}")

try:
    favorite_books_df = pd.read_csv(FAVORITE_BOOKS_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find the favorite books file: {FAVORITE_BOOKS_PATH}"
    )
except Exception as e:
    raise Exception(f"An error occurred while loading the favorite books file: {e}")


# --- Pydantic Models ---
class RecommendRequest(BaseModel):
    book_id: int
    n: int = 10


class User_RecommendRequest(BaseModel):
    user_id: str
    n: int = 10


class Recommendation(BaseModel):
    book_id: str
    label: str
    best_value: float


class ResponseFormat(BaseModel):
    status: str
    message: str
    books: List[Any]


# --- API Endpoints ---
@app.get("/popular_books", response_model=ResponseFormat)
def get_popular_books(n: int = 10):
    """
    Endpoint to get the n most popular books.
    """
    try:
        demographic_dataset = {
            "id": content_df["id"].astype(str),
            "avg_rating": content_df["avg_rating"].str.replace(",", ".").astype(float),
            "rating_count": content_df["rating_count"].astype(int),
        }
        quantile_rate = 0.9
        demographic_preprocessed_dataset = recommender.demographic_preprocess(
            demographic_dataset, quantile_rate
        )
        demographic_preprocessed_df = pd.DataFrame(demographic_preprocessed_dataset)

        popular_books = recommender.get_popular_books(demographic_preprocessed_df, n)

        return ResponseFormat(
            status="success",
            message="Popular books retrieved successfully",
            books=popular_books,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )


@app.post("/recommend/content-based", response_model=ResponseFormat)
def content_based_recommendations(req: RecommendRequest):
    """
    Endpoint to get book recommendations based on content-based filtering.
    """
    try:
        similar_books = recommender.find_similar_books(req.book_id, req.n)

        recommendations = []
        for book_id in similar_books:
            book_details = content_df[content_df["id"].astype(str) == book_id].iloc[0]
            label = book_details["category"]
            best_value = 0.9  # Placeholder, you might want to refine this

            recommendations.append(
                Recommendation(book_id=book_id, label=label, best_value=best_value)
            )

        return ResponseFormat(
            status="success",
            message="Recommendations retrieved successfully",
            books=recommendations,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )


@app.post("/recommend/colabborative", response_model=ResponseFormat)
def collaborative_recommendations(req: User_RecommendRequest):
    """
    Endpoint to get collaborative book recommendations based on user's reading preferences.
    """
    try:
        recommendations = recommender.get_collaborative_recommendations(
            req.user_id, req.n
        )
        return ResponseFormat(
            status="success",
            message="Collaborative recommendations retrieved successfully",
            books=recommendations,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )


@app.post("/recommend/personalize", response_model=ResponseFormat)
def personalized_recommendations(req: User_RecommendRequest):
    """
    Endpoint to get personalized book recommendations based on user's reading preferences.
    """
    try:
        recommendations = recommender.get_personalized_recommendations(
            req.user_id, req.n
        )
        return ResponseFormat(
            status="success",
            message="Personalized recommendations retrieved successfully",
            books=recommendations,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), books=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )
