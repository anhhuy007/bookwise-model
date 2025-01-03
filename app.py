from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Any
from book_recommender_v2 import BookRecommender, WeightedRatingModel, GradientDescentExpModel, FavouriteDataset
from fetch_data import get_books_data, get_user_info_data, get_favourite_books_data

# --- Configuration and Model Loading ---
app = FastAPI()

# Precomputed data files
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

try:
    rating_model = WeightedRatingModel()
    rating_model.load_model(RATING_N_PATH, RATING_DATASET_PATH, RATING_COUNT_THRESHOLD_PATH, RATING_AVG_MEAN_PATH)

    content_model = GradientDescentExpModel()
    content_model.load_model(CONTENT_N_PATH, CONTENT_INPUTS_PATH, CONTENT_IDS_PATH, CONTENT_LABELS_PATH, CONTENT_W_PATH)

    user_model = GradientDescentExpModel()
    user_model.load_model(USER_N_PATH, USER_INPUTS_PATH, USER_IDS_PATH, USER_LABELS_PATH, USER_W_PATH)

    favourite_dataset = FavouriteDataset()
    favourite_dataset.load_dataset(FAVOURITE_DATASET_PATH)

    # Initilize BookRecommender object
    recommender = BookRecommender(rating_model, content_model, user_model, favourite_dataset)

except Exception as e:
    print(f"Error loading precomputed data or Sentence Transformer: {e}")
    raise Exception("Error loading precomputed data or Sentence Transformer")

# Load data from database
try:
    # Fetch data from database
    books_data = get_books_data()
    user_info_data = get_user_info_data()
    favourite_books_data = get_favourite_books_data()
    
    print(f"Fetch {len(books_data)} books, {len(user_info_data)} users, and {len(favourite_books_data)} favourite books")

    # Convert to DataFrames
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

    favorite_books_df = pd.DataFrame(
        favourite_books_data,
        columns=[
            "user_id",
            "book_id",
            "added_at"
        ]
    )
except Exception as e:
    print(f"Error loading data from database: {e}")
    raise Exception("Failed to load required data from database")


# --- Pydantic Models ---
class ResponseFormat(BaseModel):
    status: str
    message: str
    books: List[Any]

class RecommendRequest(BaseModel):
    book_id: int

class User_RecommendRequest(BaseModel):
    user_id: int

class Recommendation(BaseModel):
    id: int
    title: str
    authors: str
    category: str
    best_value: float

def format_recommedation(books: list[int], best_values: np.ndarray=None) -> list[Recommendation]:
    n_books = len(books)
    if best_values is None:
        best_values = np.array([-1 for _ in range(n_books)])

    recommendations = []
    for i in range(n_books):
        # Check if book exists in dataframe
        book_matches = content_df[content_df["id"].astype(int) == int(books[i])]
        if book_matches.empty:
            print(f"Warning: Book ID {books[i]} not found in database")
            continue
        try:
            book_details = book_matches.iloc[0]
            recommendations.append(
                Recommendation(
                    id=books[i],
                    title=book_details["title"],
                    authors=book_details["authors"],
                    category=book_details["category"],
                    best_value=best_values[i]
                )
            )
        except IndexError as e:
            print(f"Error accessing book details for ID {books[i]}: {str(e)}")
            continue
    if not recommendations:
        raise ValueError("No valid recommendations found for the given book ID")
    return recommendations

# --- API Endpoints ---
@app.get("/popular_books", response_model=ResponseFormat)
def get_popular_books():
    """
    Endpoint to get the most popular books.
    """
    try:
        books, best_values = recommender.get_popular_books()
        print("Recommended books:", books)
        print("Best values:", best_values)
        return ResponseFormat(
            status="success",
            message="Popular books retrieved successfully",
            books=format_recommedation(books, best_values)
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

@app.post("/recommend/content-based", response_model=ResponseFormat)
def content_based_recommendations(req: RecommendRequest):
    """
    Endpoint to get book recommendations based on content-based filtering.
    """
    # Check if book exists in dataframe
    book_matches = content_df[content_df["id"].astype(int) == int(req.book_id)]
    if book_matches.empty:
        return ResponseFormat(
            status="fail",
            message=f"Warning: Book ID {req.book_id} not found in database",
            books=[]
        )
    book_details = book_matches.iloc[0]
    title=book_details["title"],
    category=book_details["category"]

    try:
        books, best_values = recommender.find_similar_books(int(req.book_id))
        print("Recommended books:", books)
        print("Best values:", best_values)
        return ResponseFormat(
            status="success",
            message=f"Recommendations retrieved successfully for:\nBook ID: {req.book_id}\nTitle: {title}\nCategory: {category}",
            books=format_recommedation(books, best_values)
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
    # Check if book exists in dataframe
    user_matches = user_df[user_df["id"].astype(int) == int(req.user_id)]
    if user_matches.empty:
        return ResponseFormat(
            status="fail",
            message=f"Warning: User ID {req.user_id} not found in database",
            books=[]
        )
    user_details = user_matches.iloc[0]
    goal=user_details["goal"]

    try:
        books, best_values = recommender.get_collaborative_recommendations(int(req.user_id))
        print("Recommended books:", books)
        print("Best values:", best_values)
        return ResponseFormat(
            status="success",
            message=f"Collaborative recommendations retrieved successfully for:\nUser ID: {req.user_id}\nGoal: {goal}",
            books=format_recommedation(books, best_values)
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
    # Check if book exists in dataframe
    user_matches = user_df[user_df["id"].astype(int) == int(req.user_id)]
    if user_matches.empty:
        return ResponseFormat(
            status="fail",
            message=f"Warning: User ID {req.user_id} not found in database",
            books=[]
        )
    user_details = user_matches.iloc[0]
    goal=user_details["goal"]

    try:
        books, best_values = recommender.get_personalized_recommendations(int(req.user_id))
        print("Recommended books:", books)
        print("Best values:", best_values)
        return ResponseFormat(
            status="success",
            message=f"Personalized recommendations retrieved successfully for:\nUser ID: {req.user_id}\nGoal: {goal}",
            books=format_recommedation(books, best_values)
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