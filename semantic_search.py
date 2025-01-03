import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from fetch_data import get_books_data  # Assuming fetch_data.py is in the same dir


@dataclass
class Book:
    """Book data structure"""

    id: int
    isbn13: str
    title: str
    authors: str
    description: str
    published_date: Optional[str] = None
    page_count: int = 0
    category: str = "Other"
    language: str = "en"
    avg_rating: float = 0.0
    rating_count: int = 0
    img_url: str = ""
    preview_url: Optional[str] = None

    @classmethod
    def from_db_tuple(cls, record: Tuple) -> "Book":
        """Creates a Book instance from a database tuple"""
        return cls(
            id=record[0],
            isbn13=record[1],
            title=record[2],
            authors=record[3],
            published_date=record[4],
            page_count=record[5] or 0,
            category=record[6] or "Other",
            language=record[7],
            avg_rating=float(record[8] or 0.0),
            rating_count=record[9] or 0,
            img_url=record[10],
            preview_url=record[11],
            description=record[12] or "",
        )


class BookSemanticSearch:
    """
    An improved class for performing semantic book search using FAISS
    and a pre-trained Sentence Transformer model.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        qwen_model_name: str = "Qwen/Qwen2.5-3B",
        faiss_index_path: str = "books_index.faiss",
        cache_dir: str = "cache",
    ):
        self.embedding_model_name = embedding_model_name
        self.qwen_model_name = qwen_model_name
        self.faiss_index_path = faiss_index_path
        self.cache_dir = cache_dir

        # Initialize components
        self.embedding_model = None
        self.qwen_tokenizer = None
        self.qwen_model = None
        self.index = None
        self.books = []  # Store books after fetching

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_books(self) -> List[Book]:
        """Fetches books using the provided get_books_data function"""
        try:
            db_records = get_books_data()
            self.books = [Book.from_db_tuple(record) for record in db_records]
            self.logger.info(f"Fetched {len(self.books)} books from database")
            return self.books
        except Exception as e:
            self.logger.error(f"Error fetching books: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Cleans text by removing/replacing special characters"""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        text = text.strip()
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def format_authors(self, authors: str) -> str:
        """Formats the authors field to ensure proper quoting for multiple authors"""
        if pd.isna(authors) or authors is None:
            return ""
        authors = str(authors)
        return f'"{authors}"' if "," in authors else authors

    def prepare_text_for_embedding(self, book: Book) -> str:
        """Prepares book text for embedding"""
        title = self.clean_text(book.title)
        authors = self.format_authors(book.authors)
        description = self.clean_text(book.description)
        category = self.clean_text(book.category)

        return f"{title} {authors} {category} {description}"

    def initialize_embedding_model(self):
        """Initializes the Sentence Transformer embedding model"""
        if self.embedding_model is None:
            self.logger.info(
                f"Initializing embedding model: {self.embedding_model_name}"
            )
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def generate_embeddings(self, books: List[Book]) -> np.ndarray:
        """Generates embeddings for the books"""
        self.initialize_embedding_model()

        texts = [self.prepare_text_for_embedding(book) for book in books]
        self.logger.info("Generating embeddings...")

        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        return embeddings

    def initialize_faiss_index(self, embeddings: np.ndarray):
        """Initializes the FAISS index with the given embeddings"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.logger.info("FAISS index initialized")

    def save_faiss_index(self):
        """Saves the FAISS index to a file"""
        if self.index is not None:
            index_path = os.path.join(self.cache_dir, self.faiss_index_path)
            faiss.write_index(self.index, index_path)
            self.logger.info(f"FAISS index saved to {index_path}")

    def load_faiss_index(self) -> bool:
        """Loads the FAISS index from a file"""
        index_path = os.path.join(self.cache_dir, self.faiss_index_path)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.logger.info(f"FAISS index loaded from {index_path}")
            return True
        return False

    def build_index(self):
        """Builds the FAISS index from scratch using fetched books"""
        # Fetch books
        self.books = self.fetch_books()
        if not self.books:
            raise ValueError("No books available to build index")

        # Generate embeddings
        embeddings = self.generate_embeddings(self.books)

        # Initialize and save index
        self.initialize_faiss_index(embeddings)
        self.save_faiss_index()

    def search_books(
        self, query: str, k: int = 5, similarity_threshold: float = 0.1
    ) -> List[Tuple[Book, float]]:
        """Searches for books similar to the query"""
        if not self.books:
            raise ValueError("No books available for search")

        # Ensure embedding model is initialized before use
        if self.embedding_model is None:
            self.initialize_embedding_model()

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype("float32"), k)

        # Filter and prepare results
        results = []
        for idx, (distance, book_idx) in enumerate(zip(distances[0], indices[0])):
            similarity_score = float(distance)
            if similarity_score >= similarity_threshold:
                results.append((self.books[book_idx], similarity_score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def find_similar_books(self, query: str, k: int = 5) -> List[dict]:
        """Main method to find books similar to the query"""
        try:
            # Load or build index if necessary
            if self.index is None:
                if not self.load_faiss_index():
                    print("Building index...")
                    self.build_index()
                elif not self.books:  # If index was loaded but books weren't
                    self.books = self.fetch_books()

            # Ensure embedding model is initialized
            if self.embedding_model is None:
                self.initialize_embedding_model()

            # Search for similar books
            results = self.search_books(query, k)

            # Format results
            formatted_results = []
            for book, score in results:
                formatted_results.append(
                    {
                        "id": book.id,
                        "title": book.title,
                        "authors": book.authors,
                        "category": book.category,
                        "rating": book.avg_rating,
                        "similarity_score": f"{score:.2f}",
                        "view_count": book.rating_count,
                        "img_url": book.img_url,
                    }
                )

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error during book search: {e}")
            raise
