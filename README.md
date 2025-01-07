# Book Recommendation Project

This repository contains models to recommend books for users, including:

1.  **Demographic Filtering Model:** Recommends books based on aggregated user data, such as popularity and average ratings.
2.  **Collaborative Filtering Model:** Identifies users with similar tastes and suggests books they have enjoyed.
3.  **Content-Based Filtering Model:** Recommends books that share similar themes or genres with the user's previously read or liked books.
4.  **Semantic Search Model:** Leverages Natural Language Processing (NLP) and vector databases to understand the meaning and context of user queries and book descriptions, providing a more refined search experience.

---


## Getting Started
1.  **Install FastAPI:**
    ```bash
    pip install "fastapi[standard]"
    ```
    
---

## To Run the Project

1.  **Run the preprocessing script:**
    ```bash
    python preprocessed_data.py
    ```
    This script will prepare the data for the models.
2.  **Run the server with:**
    ```bash
    fastapi dev app.py
    ```
3.  **Access the API:** After launching the server, you can access the interactive API documentation at `http://127.0.0.1:8000/docs` to explore the available endpoints.

---

## For more information
  Read our report: [Book Recommendation Report](reports/Book_Recommend.pdf)


