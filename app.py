from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Any, Optional
from book_recommender import BookRecommender
from fetch_data import get_books_data, get_user_info_data, get_favourite_books_data
from semantic_search import BookSemanticSearch  # Import the model class

# --- Configuration and Model Loading ---
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",  # Add your frontend's origin(s)
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict methods if needed e.g., ["GET", "POST"]
    allow_headers=["*"],  # You can restrict headers if needed
)
# --- End of CORS Setup ----

# Precomputed data files
CONTENT_EMBEDDINGS_PATH = "./preprocessed_data/content_embeddings.pkl"
CONTENT_LABELS_PATH = "./preprocessed_data/content_labels.pkl"
COTENT_WEIGHTS_PATH = "./preprocessed_data/content_w.pkl"  #
CONTENT_PREPROCESSED_INPUT_PATH = (
    "./preprocessed_data/content_preprocessed_input.pkl"  #
)
USER_EMBEDDINGS_PATH = "./preprocessed_data/user_embeddings.pkl"
USER_LABELS_PATH = "./preprocessed_data/user_labels.pkl"
USER_WEIGHTS_PATH = "./preprocessed_data/user_w.pkl"
USER_PREPROCESSED_INPUT_PATH = "./preprocessed_data/user_preprocessed_input.pkl"
FAVORITE_BOOKS_PREPROCESSED_INPUT_PATH = (
    "./preprocessed_data/favorite_preprocessed_input.pkl"
)
SENTENCE_TRANSFORMER_PATH = "./preprocessed_data/sentence_transformer.pkl"

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

# Load data from database
try:
    # Fetch data from database
    books_data = get_books_data()
    user_info_data = get_user_info_data()
    favourite_books_data = get_favourite_books_data()

    print(
        f"Fetch {len(books_data)} books, {len(user_info_data)} users, and {len(favourite_books_data)} favourite books"
    )

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
        ],
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
        ],
    )

    favorite_books_df = pd.DataFrame(
        favourite_books_data, columns=["user_id", "book_id", "added_at"]
    )

    # Update recommender with database data
    recommender.books_data = content_df
    recommender.user_info_data = user_df
    recommender.favourite_books_data = favorite_books_df

except Exception as e:
    print(f"Error loading data from database: {e}")
    raise Exception("Failed to load required data from database")


# --- Pydantic Models ---
class RecommendRequest(BaseModel):
    book_id: int
    n: int = 10


class User_RecommendRequest(BaseModel):
    user_id: str
    n: int = 10


class Recommendation(BaseModel):
    id: str
    title: str
    authors: str
    category: str
    best_value: float


class ResponseFormat(BaseModel):
    status: str
    message: str
    books: List[Any]


class BookSearchResponse(BaseModel):
    id: int
    title: str
    authors: str
    category: str
    rating: float
    similarity_score: str
    preview_url: Optional[str] = None
    img_url: str


class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(5, description="Number of top similar books to return")


# --- API Endpoints ---
@app.get("/popular_books", response_model=ResponseFormat)
def get_popular_books(n: int = 10):
    """
    Endpoint to get the n most popular books.
    """
    try:
        demographic_dataset = {
            "id": content_df["id"].astype(str),
            "avg_rating": content_df["avg_rating"].astype(float),
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
            # Check if book exists in dataframe
            book_matches = content_df[content_df["id"].astype(str) == str(book_id)]
            if book_matches.empty:
                print(f"Warning: Book ID {book_id} not found in database")
                continue

            try:
                book_details = book_matches.iloc[0]
                recommendations.append(
                    Recommendation(
                        id=book_id,
                        title=book_details["title"],
                        authors=book_details["authors"],
                        category=book_details["category"],
                        best_value=0.9,
                    )
                )
            except IndexError as e:
                print(f"Error accessing book details for ID {book_id}: {str(e)}")
                continue

        if not recommendations:
            raise ValueError("No valid recommendations found for the given book ID")

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

book_search_engine = BookSemanticSearch()

@app.on_event("startup")
async def startup_event():
    """Initialize book search engine, and load embedding and FAISS model."""
    print("Starting app...")
    try:
        if not book_search_engine.load_faiss_index():
            print("Building the index...This will take a while.")
            book_search_engine.build_index()
        elif not book_search_engine.books:
            book_search_engine.fetch_books()
        book_search_engine.initialize_embedding_model()  # Ensure embedding model initialized
    except Exception as e:
        print(f"Error initializing the app: {e}")
        raise


# API dependency
def get_book_search_engine() -> BookSemanticSearch:
    return book_search_engine


# API endpoint
@app.post("/search", response_model=List[BookSearchResponse])
async def search_books_endpoint(
    search_query: SearchQuery,
    book_search: BookSemanticSearch = Depends(get_book_search_engine),
):
    try:
        results = book_search.find_similar_books(
            search_query.query, k=search_query.top_k
        )
        
        print(f"Search query: {search_query.query}")
        for idx, result in enumerate(results):
            print(f"Result {idx+1}: {result['title']} by {result['authors']} - {result['similarity_score']}")
        
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
