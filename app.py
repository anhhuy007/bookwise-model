from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any
from book_recommender import BookRecommender

# --- Configuration and Model Loading ---
app = FastAPI()

# Paths to your precomputed data files
EMBEDDINGS_PATH = "./preprocessed_data/embeddings.pkl"
LABELS_PATH = "./preprocessed_data/labels.pkl"
WEIGHTS_PATH = "./preprocessed_data/content_w.pkl"
PREPROCESSED_INPUT_PATH = "./preprocessed_data/preprocessed_input.pkl"
SENTENCE_TRANSFORMER_PATH = "./preprocessed_data/sentence_transformer.pkl"
DATASET_PATH = "./dataset/content_based_data.csv"

try:
    recommender = BookRecommender(sentence_transformer_path=SENTENCE_TRANSFORMER_PATH)
    recommender.load_precomputed_data(
        embeddings_path=EMBEDDINGS_PATH,
        labels_path=LABELS_PATH,
        weights_path=WEIGHTS_PATH,
        preprocessed_input_path=PREPROCESSED_INPUT_PATH,
    )
except Exception as e:
    print(f"Error loading precomputed data or Sentence Transformer: {e}")
    raise  # Re-raise the exception to stop the application

try:
    content_df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the dataset file: {DATASET_PATH}")
except Exception as e:
    raise Exception(f"An error occurred while loading the dataset: {e}")


# --- Pydantic Models ---
class RecommendRequest(BaseModel):
    book_id: int
    n: int = 10


class Recommendation(BaseModel):
    book_id: str
    label: str
    best_value: float


class ResponseFormat(BaseModel):
    status: str
    message: str
    data: List[Any]  # Use Any if the data structure can vary


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
            data=popular_books,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), data=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )


@app.post("/recommend", response_model=ResponseFormat)
def recommend(req: RecommendRequest):
    """
    Endpoint to get book recommendations based on a book_id.
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
            data=recommendations,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=ResponseFormat(
                status="error", message=str(e), data=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResponseFormat(
                status="error", message=str(e), data=[]
            ).dict(),  # Convert to dictionary for HTTPException
        )
